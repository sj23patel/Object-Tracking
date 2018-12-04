'''
    File name         : object_tracking.py
    File Description  : Multi Object Tracker Using Kalman Filter
                        and Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import cv2
import copy
import numpy as np
import time 
import keyboard
from tracker import Tracker
#from kalman_filter_backup import KalmanFilter
from pydarknet import Detector, Image
from video_demo import video_demo as vid


def main():
    """Main function for multi object tracking
    Usage:
        $ python2.7 objectTracking.py
    Pre-requisite:
        - Python2.7
        - Numpy
        - SciPy
        - Opencv 3.0 for Python
    Args:
        None
    Return:
        None
    """


    '''
    options = {
            'model': 'cfg/tiny-yolo-voc-1c.cfg',
            'load': 4000,
            'threshold': 0.15,
            'gpu': 1.0
    }

    tfnet = TFNet(options)
    '''
    net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,
                    bytes("cfg/coco.data", encoding="utf-8"))

    # Create opencv video capture object
    cap = cv2.VideoCapture('mansiroad_trimmed.mp4')
    #length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    '''
    # Create opencv video capture object
    cap = cv2.VideoCapture('data/TrackingBugs.mp4')

    # Create Object Detectorx
    detector = Detectors()
    '''
    # Create Object Tracker
    tracker = Tracker(25, 60, 1000, 10)

    # Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False
    frame_array=[]
    first=0
    # Infinite loop to process video frames
    size=(1920,1080)
    c=0
    #print(length)
    #while(c<=length-1):
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            #print('ret',ret)
            # Make copy of original frame
            orig_frame = copy.copy(frame)

            # Skip initial frames that display logo
            '''
            if (skip_frame_count < 15):
                skip_frame_count += 1
                continue
            '''
            #print('ssss')
            # Detect and return centeroids of the objects in the frame
            centers = vid.detect(ret,frame,net)#,tfnet)
            print("centers :",centers)
            # If centroids are detected then track them
            if (len(centers) > 0):
                first=1
                # Track object using Kalman Filter
                tracker.Update(centers,first)

                # For identified object tracks draw tracking line
                # Use various colors to indicate different track_id
                print('NUM OF OBJECTS : ',len(tracker.tracks))
                for i in range(len(tracker.tracks)):
                    if (len(tracker.tracks[i].trace) > 1):
                        #print('NUM OF OBJECTS : ',tracker.tracks[i].trace)
                        for j in range(len(tracker.tracks[i].trace)-1):
                            # Draw trace line
                            x1 = tracker.tracks[i].trace[j][0][0]
                            y1 = tracker.tracks[i].trace[j][1][0]
                            x2 = tracker.tracks[i].trace[j+1][0][0]
                            y2 = tracker.tracks[i].trace[j+1][1][0]
                            clr = tracker.tracks[i].track_id % 9
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                     track_colors[clr], 2)

                # Display the resulting tracking frame
                #cv2.imshow('Tracking', frame)
            elif first==1:
                tracker.Update(centers,0)
                print('NUM OF OBJECTSno : ',len(tracker.tracks))
                for i in range(len(tracker.tracks)):
                    if (len(tracker.tracks[i].trace) > 1):
                        print('NUM OF OBJECTSnononono : ',len(tracker.tracks[i].trace),)
                        print('trace : ',tracker.tracks[i].trace[len(tracker.tracks[i].trace)-1],)
                        
                        for j in range(len(tracker.tracks[i].trace)-1):
                            # Draw trace line
                            x1 = tracker.tracks[i].trace[j][0][0]
                            y1 = tracker.tracks[i].trace[j][1][0]
                            x2 = tracker.tracks[i].trace[j+1][0][0]
                            y2 = tracker.tracks[i].trace[j+1][1][0]
                            clr = tracker.tracks[i].track_id % 9

                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                     track_colors[clr], 2)

            height, width, layers = frame.shape
            size = (width,height)
            frame_array.append(frame)
            cv2.imshow('ss',frame)
            '''
            # Display the original frame
            #cv2.imshow('Original', orig_frame)
            if keyboard.is_pressed('q'):# 'q' key has been pressed, exit program.
                break
            # Slower the FPS
            cv2.waitKey(50)

            # Check for key strokes
            k = cv2.waitKey(50) & 0xff
            if k == 27:  # 'esc' key has been pressed, exit program.
                break
            if k == 112:  # 'p' has been pressed. this will pause/resume the code.
                pause = not pause
                if (pause is True):
                    print("Code is paused. Press 'p' to resume..")
                    while (pause is True):
                        # stay in this loop until
                        key = cv2.waitKey(30) & 0xff
                        if key == 112:
                            pause = False
                            print("Resume code..!!")
                            break
            '''
            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key == ord('q'):
                break

            # Take screenshot
            if key == ord('s'):
                cv2.imwrite('frame_{}.jpg'.format(time.time()), frame)

            c+=1
    except:
        print('Video Ended')
    # When everything done, release the capture
    finally:
        #out = cv2.VideoWriter('result2.mp4',cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), size)
        #print('11')
        #for i in range(len(frame_array)):
            # writing to a image array
            #cv2.imshow('ff',frame_array[i])
            # writing to a image array
            #out.write(frame_array[i])
        #out.release()
        
        #out.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    main()
