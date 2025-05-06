import cv2
import os
import time

COLLECTED_DATA_PATH = './../PSL_Collected_Data'
LABELS = ['A', 'B', 'C', 'CH', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'SZ', 'T', 'U', 'W', 'Y','Z']
NUMBER_OF_IMG_PER_LABEL = 50

def capture_images(label):
    print(f'Currently collecting label: {label}')
    os.makedirs(f'{COLLECTED_DATA_PATH}/{label}', exist_ok=True)
    cap = cv2.VideoCapture(0)
    time.sleep(3)

    for image_number in range(NUMBER_OF_IMG_PER_LABEL):
        ret, frame = cap.read()
        if not ret:
            print("Failed")
            break

        print(f'Label: {label} | Image: {image_number}/{NUMBER_OF_IMG_PER_LABEL - 1}')

        file_name = f'{COLLECTED_DATA_PATH}/{label}/{label}_{image_number}.jpg'
        cv2.imwrite(file_name, frame)
        cv2.imshow('PJM Capture', frame)
        time.sleep(0.3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()
    cv2.destroyAllWindows()


for letter in LABELS:
    capture_images(letter)
