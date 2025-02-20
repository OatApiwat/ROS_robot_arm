import cv2
import time

def main():
    # เปิดกล้อง
    cap = cv2.VideoCapture(0)

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
    cv2.resizeWindow(window_name, 800, 600)

    while True:
        # อ่านเฟรมจากกล้อง
        ret, frame = cap.read()
        
        if not ret:
            print("ไม่สามารถอ่านเฟรมจากกล้องได้")
            break

        # เพิ่มจำนวนเฟรม
        frame_count += 1

        # คำนวณเฟรมเรตทุกๆ 1 วินาที
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time  # คำนวณเฟรมเรต
            frame_count = 0  # รีเซ็ตจำนวนเฟรม
            start_time = time.time()  # รีเซ็ตเวลา

        # แสดงเฟรมเรตบนภาพ
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ปรับขนาดเฟรมให้ตรงกับขนาดหน้าต่าง
        frame_resized = cv2.resize(frame, (800, 600))

        # แสดงภาพ
        cv2.imshow(window_name, frame_resized)

        # ออกจากลูปเมื่อกดปุ่ม 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ปิดกล้องและหน้าต่างทั้งหมด
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
