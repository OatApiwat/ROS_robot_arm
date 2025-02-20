#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray
import cv2
import numpy as np
import time
import signal,os
import math
rospy.init_node('camera_node',anonymous=True)
pub_camera = rospy.Publisher('/cam_to_servo_topic', Float64MultiArray, queue_size=10)

# ฟังก์ชันที่จะเรียกเมื่อมีการกด Ctrl+C
def signal_handler(sig, frame):
    print('Ctrl+C for kill program')
    os._exit(0)  # หยุดโปรแกรม

def pixel_to_meter(camera_height, image_width, fov=90):
    # คำนวณความกว้างจริงของภาพในโลกจริง
    image_height_px = 480
    real_width = 2 * camera_height * math.tan(math.radians(fov / 2))
    
    # คำนวณอัตราส่วนการแปลงพิกเซลเป็นเมตร
    pixel_per_meter = real_width / image_height_px
    
    return pixel_per_meter*image_width/2

def get_object_properties(contour, area_min_threshold, area_max_threshold):
    area = cv2.contourArea(contour)
    
    if area < area_min_threshold or area > area_max_threshold:
        return None
    
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    center = (int(rect[0][0]), int(rect[0][1]))
    

    # หา index ของด้านที่ยาวที่สุด
    # จุดที่ 1
    point1 = box[0]
    # จุดที่ 2
    point2 = box[1]
    # จุดที่ 4
    point4 = box[3]
    # คำนวณระยะทางระหว่างจุดที่ 1 และจุดที่ 2
    distance_12 = np.linalg.norm(np.array(point2) - np.array(point1))
    
    # คำนวณระยะทางระหว่างจุดที่ 1 และจุดที่ 4
    distance_14 = np.linalg.norm(np.array(point4) - np.array(point1))
    
    # คืนค่าระยะทางที่มากที่สุดระหว่างสองระยะ
    max_distance = max(distance_12, distance_14)
    min_distance = min(distance_12, distance_14)
    
    # คำนวณความชันของด้านที่ยาวที่สุด
    if max_distance == distance_12:
        dx = (point2[0] - point1[0])  # ความแตกต่างในแกน x
        dy = (point2[1] - point1[1])   # ความแตกต่างในแกน y
    else:
        dx = (point4[0] - point1[0])  # ความแตกต่างในแกน x
        dy = (point4[1] - point1[1])   # ความแตกต่างในแกน y
        
    # หามุมที่กระทำกับแกน X โดยใช้ arctan
    angle_rad = np.arctan2(dy, dx)  # มุมในหน่วย radians
    angle_deg = np.degrees(angle_rad)  # แปลงมุมเป็น degrees
    real_width = pixel_to_meter(0.14, min_distance,70)
    return center, box, area,angle_deg,real_width

def camera(cap,window_name):
    global frame_count,start_time,fps
    while True:
        ret, img = cap.read()
        if not ret:
            break
                
        # อ่านค่าจาก trackbars
        C_MEAN_Threshold = cv2.getTrackbarPos('C_MEAN_Threshold', 'Object Setting')
        C_GAUSSIAN_Threshold = cv2.getTrackbarPos('C_GAUSSIAN_Threshold', 'Object Setting')
        area_min_threshold = cv2.getTrackbarPos('Area min_Threshold', 'Object Setting')
        area_max_threshold = cv2.getTrackbarPos('Area max_Threshold', 'Object Setting')
        mean_c_threshold = cv2.getTrackbarPos('MEAN_C Threshold', 'Object Setting')
        gaussian_c_threshold = cv2.getTrackbarPos('GAUSSIAN_C Threshold', 'Object Setting')
        canny_min_threshold = cv2.getTrackbarPos('Canny_min Threshold', 'Object Setting')
        canny_max_threshold = cv2.getTrackbarPos('Canny_max Threshold', 'Object Setting')
        # แปลงค่า threshold ให้เป็นเลขคี่
        mean_c_threshold = max(mean_c_threshold * 2 + 1,3)
        gaussian_c_threshold = max(gaussian_c_threshold * 2 + 1,3)

        # เพิ่มจำนวนเฟรม
        frame_count += 1
        # คำนวณเฟรมเรตทุกๆ 1 วินาที
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time  # คำนวณเฟรมเรต
            frame_count = 0  # รีเซ็ตจำนวนเฟรม
            start_time = time.time()  # รีเซ็ตเวลา
        # แสดงเฟรมเรตบนภาพ
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # ปรับขนาดเฟรมให้ตรงกับขนาดหน้าต่าง
        frame_resized = cv2.resize(img, (640, 480))
        #แปลงภาพ
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        th_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, mean_c_threshold, C_MEAN_Threshold)
        th_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, gaussian_c_threshold, C_GAUSSIAN_Threshold)
        gblur_mean = cv2.GaussianBlur(th_mean,(3,3),0)
        gblur_gaussian = cv2.GaussianBlur(th_gaussian,(3,3),0)
        canny_mean = cv2.Canny(gblur_mean,canny_min_threshold,canny_max_threshold)
        canny_gaussian = cv2.Canny(gblur_gaussian,canny_min_threshold,canny_max_threshold)
        contours_mean,_ = cv2.findContours(canny_mean,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contours_gaussian,_ = cv2.findContours(canny_gaussian,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        output_image_mean = np.zeros_like(frame_resized)
        output_image_gaussian = np.zeros_like(frame_resized)
        cv2.drawContours(output_image_mean,contours_mean,-1,(0,0,255),2)
        cv2.drawContours(output_image_gaussian,contours_gaussian,-1,(0,0,255),2)

        iter = 0
        center_picture = [640/2,480/2]
        object_data = []
        for contour in contours_mean:
            properties = get_object_properties(contour, area_min_threshold, area_max_threshold)
            if properties:
                    center, box, area,angle_deg,real_width = properties
                    iter+=1
                    # center_filtered = moving_average(center, center_values, window_size)
                    # area_filtered = moving_average(area, area_values, window_size)
                    # lengths_filtered = moving_average(side_lengths_meters, lengths_values, window_size)
                    # corners_filtered = moving_average(box, corner_values, window_size)
                    # angle_deg_filtered = moving_average(angle_deg, angle_values, window_size)

                    center_filtered = center
                    area_filtered = area 
                    corners_filtered = box

                    # แสดงผลข้อมูล
                    # print(f"iter: {iter},center: {center_filtered}, corners: {corners_filtered.tolist()}, angle_deg: {angle_deg},real_width:{real_width}")
                    # วาดกรอบและจุดกึ่งกลางที่กรองแล้ว
                    cv2.drawContours(output_image_mean, [np.int0(corners_filtered)], 0, (0, 255, 0), 2)
                    cv2.circle(output_image_mean, (int(center_filtered[0]), int(center_filtered[1])), 5, (255, 0, 0), -1)
                    # แสดงผลข้อมูล
                    if iter %2 ==0:
                    #     print(f"iter: {iter/2},center: {center_filtered}, corners: {corners_filtered.tolist()}, angle_deg: {angle_deg},real_width:{real_width}")
                    # คำนวณระยะห่างระหว่าง center_filtered และ center_picture
                        distance = np.linalg.norm(np.array(center_filtered) - np.array(center_picture))
                        
                        # เก็บข้อมูลในลิสต์
                        object_data.append((distance, iter/2, center_filtered, corners_filtered.tolist(), angle_deg, real_width))
        # เรียงลำดับ object_data ตามระยะห่าง
        object_data.sort(key=lambda x: x[0])
        iter_sort = 0
        for distance, iter_half, center_filtered, corners_filtered, angle_deg, real_width in object_data:
            iter_sort +=1
            print(f"iter: {iter_sort}, center: {center_filtered}, corners: {corners_filtered}, angle_deg: {angle_deg}, real_width: {real_width}, distance: {distance}")
            text = f"Iter: {iter_sort}, Dist: {distance:.2f}"
            cv2.putText(output_image_mean, text, 
                        (int(center_filtered[0]), int(center_filtered[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)   
        # แสดงภาพ
        cv2.imshow(window_name, frame_resized)
        cv2.imshow("Gray",gray)
        cv2.imshow('ADAPTIVE_THRESH_MEAN_C', th_mean)
        cv2.imshow('ADAPTIVE_THRESH_GAUSSIAN_C', th_gaussian)
        cv2.imshow('gblur_mean', gblur_mean)
        cv2.imshow('gblur_gaussian', gblur_gaussian)
        cv2.imshow('canny_mean', canny_mean)
        cv2.imshow('canny_gaussian', canny_gaussian)
        cv2.imshow('output_image_mean', output_image_mean)
        cv2.imshow('output_image_gaussian', output_image_gaussian)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main(cap):
    global fps,frame_count,start_time
    signal.signal(signal.SIGINT, signal_handler)
    # ตั้งค่าความละเอียด
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ตัวแปรสำหรับการคำนวณเฟรมเรต
    fps = 0
    frame_count = 0
    start_time = time.time()
    # กำหนดชื่อหน้าต่าง
    window_name = 'Camera Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    while not rospy.is_shutdown():
        camera(cap,window_name)
        rospy.spin()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        cap = cv2.VideoCapture(0)  
        # Create trackbars for object setting
        cv2.namedWindow('Object Setting')
        cv2.createTrackbar('Area min_Threshold', 'Object Setting', 5000, 100000, lambda x: None)
        cv2.createTrackbar('Area max_Threshold', 'Object Setting', 150000, 500000, lambda x: None)
        cv2.createTrackbar('C_MEAN_Threshold', 'Object Setting', 100, 1111, lambda x: None)
        cv2.createTrackbar('C_GAUSSIAN_Threshold', 'Object Setting', 120, 1111, lambda x: None)
        cv2.createTrackbar('MEAN_C Threshold', 'Object Setting', 400, 1111, lambda x: None)
        cv2.createTrackbar('GAUSSIAN_C Threshold', 'Object Setting', 120, 1111, lambda x: None)
        cv2.createTrackbar('Canny_min Threshold', 'Object Setting', 120, 255, lambda x: None)
        cv2.createTrackbar('Canny_max Threshold', 'Object Setting', 120, 255, lambda x: None)
        main(cap)
    except rospy.ROSInterruptException:
        cap.release()
        cv2.destroyAllWindows()