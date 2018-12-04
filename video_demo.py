import time
import argparse
from pydarknet import Detector, Image
import cv2

class video_demo:
#if __name__ == "__main__":
    def detect(r,frame,net):
        '''
        parser = argparse.ArgumentParser(description='Process a video.')
        parser.add_argument('path', metavar='video_path', type=str,
                            help='Path to source video')

        args = parser.parse_args()
        print("Source Path:", args.path)
        cap = cv2.VideoCapture(args.path)
        '''
        average_time = 0
        centroids=[]
        #while True:
        #r, frame = cap.read()
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead
            #print('ccccccc')
            dark_frame = Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            average_time = average_time * 0.8 + (end_time-start_time) * 0.2

            print("FPS :", 1/(end_time-start_time), "Average_Time :", average_time)

            for cat, score, bounds in results:
                if cat.decode("utf-8")=="car" or cat.decode("utf-8")=="truck" or cat.decode("utf-8")=="vehicle":
                #if cat.decode("utf-8")=="sports ball":
                    x, y, w, h = bounds
                    cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
                    cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
                    centroids.append((x,y))
            #cv2.imshow("preview", frame)
        return centroids
        '''
        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break
        '''