import cv2

def run_analysis():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video")

    while(cap.isOpened()):
        status, frame = cap.read()
        if status:
            cv2.imshow('frame', frame)
            # do_stuff_with_frame(frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_analysis()