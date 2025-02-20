import cv2
import numpy as np
import sys
import time

prv_time = time.time()

# ฟังก์ชันเพื่อสร้าง trackbars
def nothing(x):
    pass

def main(cap):
    global prv_time
    while True:
        ret, img = cap.read()
        if not ret:
            break
        width = 640  # ความกว้างที่ต้องการ
        height = 480  # ความสูงที่ต้องการ
        img = cv2.resize(img, (width, height))
        # อ่านค่าจาก trackbars
        # area_min_threshold = cv2.getTrackbarPos('Area min_Threshold', 'Object Setting')
        # area_max_threshold = cv2.getTrackbarPos('Area max_Threshold', 'Object Setting')
        # mean_c_threshold = cv2.getTrackbarPos('MEAN_C Threshold', 'Object Setting')
        # gaussian_c_threshold = cv2.getTrackbarPos('GAUSSIAN_C Threshold', 'Object Setting')
        # canny_min_threshold = cv2.getTrackbarPos('Canny_min Threshold', 'Object Setting')
        # canny_max_threshold = cv2.getTrackbarPos('Canny_max Threshold', 'Object Setting')

        # แปลงค่า threshold ให้เป็นเลขคี่
        # mean_c_threshold = max(mean_c_threshold * 2 + 1,3)
        # gaussian_c_threshold = max(gaussian_c_threshold * 2 + 1,3)

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # แปลงภาพ
        # th_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, mean_c_threshold, 1)
        # th_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, gaussian_c_threshold, 1)
        # gblur_mean = cv2.GaussianBlur(th_mean,(3,3),0)
        # gblur_gaussian = cv2.GaussianBlur(th_gaussian,(3,3),0)
        # canny_mean = cv2.Canny(gblur_mean,canny_min_threshold,canny_max_threshold)
        # canny_gaussian = cv2.Canny(gblur_gaussian,canny_min_threshold,canny_max_threshold)
        # contours_mean,_ = cv2.findContours(canny_mean,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        # contours_gaussian,_ = cv2.findContours(canny_gaussian,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        # output_image = np.zeros_like(img)
        # cv2.drawContours(output_image,contours_mean,-1,(0,255,0),2)
        # cv2.drawContours(th_gaussian,contours_mean,-1,(0,255,0),1)

        # แสดงภาพ
        cv2.imshow('Original', img)
        print(time.time()-prv_time)
        prv_time = time.time()
        # cv2.imshow('ADAPTIVE_THRESH_MEAN_C', th_mean)
        # cv2.imshow('ADAPTIVE_THRESH_GAUSSIAN_C', th_gaussian)
        # cv2.imshow('gblur_mean', gblur_mean)
        # cv2.imshow('gblur_gaussian', gblur_gaussian)
        # cv2.imshow('canny_mean', canny_mean)
        # cv2.imshow('canny_gaussian', canny_gaussian)
        # cv2.imshow('output_image', output_image)

        # จัดการการออกจากลูป
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # ใช้กล้องเว็บแคม
    # cap  = cv2.imread("image/ant.jpg")

    # สร้างหน้าต่าง
    # cv2.namedWindow('Object Setting')
    # cv2.createTrackbar('Area min_Threshold', 'Object Setting', 5000, 10000, nothing)
    # cv2.createTrackbar('Area max_Threshold', 'Object Setting', 150000, 500000, nothing)
    # cv2.createTrackbar('MEAN_C Threshold', 'Object Setting', 120, 1111, nothing)
    # cv2.createTrackbar('GAUSSIAN_C Threshold', 'Object Setting', 120, 1111, nothing)
    # cv2.createTrackbar('Canny_min Threshold', 'Object Setting', 120, 255, nothing)
    # cv2.createTrackbar('Canny_max Threshold', 'Object Setting', 120, 255, nothing)

    try:
        main(cap)
    except KeyboardInterrupt:
        print("\nProgram stopped by user with Ctrl+C.")
        sys.exit(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()