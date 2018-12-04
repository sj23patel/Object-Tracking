'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
from kalman_filter_backup import KalmanFilter
from common import dprint
from scipy.optimize import linear_sum_assignment
from ukf import UKF
import time
import cv2
class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = UKF(prediction)#,self.iterate_x)  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path

class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount
        self.ls=[]
    def Update(self, detections,flag):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """
        assignment = []
        
        if flag==1:
            # Create tracks if no tracks vector found
            if (len(self.tracks) == 0):
                for i in range(len(detections)):
                    print('detections : ',detections)
                    track = Track(detections[i], self.trackIdCount)
                    self.trackIdCount += 1
                    self.tracks.append(track)

            # Calculate cost using sum of square distance between
            # predicted vs detected centroids
            N = len(self.tracks)
            M = len(detections)
            cost = np.zeros(shape=(N, M))   # Cost matrix
            print ('N: ',N)
            print('M: ',M)
            
            for i in range(len(self.tracks)):
                for j in range(len(detections)):
                    try:
                        diff1 = np.subtract(self.tracks[i].prediction[0] , detections[j][0])
                        diff2 = np.subtract(self.tracks[i].prediction[1] , detections[j][1])
                        diff=np.array([[diff1],[diff2]])
                        distance = np.sqrt(diff[0][0]*diff[0][0] +
                                               diff[1][0]*diff[1][0])
                        cost[i][j] = distance
                        #print("differece: ",diff,"prediction: ",self.tracks[i].prediction,"detections: ",detections[j])
                        #print("cost: i: ",i," j: ",j,": ",cost[i][j])
                    except:
                        pass

            # Let's average the squared ERROR
            cost = (0.5) * cost
            # Using Hungarian Algorithm assign the correct detected measurements
            # to predicted tracks
            
            for _ in range(N):
                assignment.append(-1)
            row_ind, col_ind = linear_sum_assignment(cost)
            for i in range(len(row_ind)):
                assignment[row_ind[i]] = col_ind[i]
            # Identify tracks with no assignment, if any
            un_assigned_tracks = []
            for i in range(len(assignment)):
                if (assignment[i] != -1):
                    # check for cost distance threshold.
                    # If cost is very high then un_assign (delete) the track
                    if (cost[i][assignment[i]] > self.dist_thresh):
                        assignment[i] = -1
                        un_assigned_tracks.append(i)
                    pass
                else:
                    self.tracks[i].skipped_frames += 1
            print ('Assignment after thresholding: ', assignment)
            # If tracks are not detected for long time, remove them
            del_tracks = []
            for i in range(len(self.tracks)):
                if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                    del_tracks.append(i)
            if len(del_tracks) > 0:  # only when skipped frame exceeds max
                for id in del_tracks:
                    if id < len(self.tracks):
                        del self.tracks[id]
                        del assignment[id]
                    else:
                        dprint("ERROR: id is greater than length of tracks")

            # Now look for un_assigned detects
            un_assigned_detects = []
            for i in range(len(detections)):
                    if i not in assignment:
                        un_assigned_detects.append(i)

            # Start new tracks
            if(len(un_assigned_detects) != 0):
                for i in range(len(un_assigned_detects)):
                    track = Track(detections[un_assigned_detects[i]],
                                  self.trackIdCount)
                    self.trackIdCount += 1
                    self.tracks.append(track)

            # Update KalmanFilter state, lastResults and tracks trace
            for i in range(len(assignment)):
                print('before only prediction: ', self.tracks[i].KF.x)
                self.tracks[i].KF.predict(0.05)
                print('after only prediction: ', self.tracks[i].KF.x)
                if(assignment[i] != -1):
                    self.tracks[i].skipped_frames = 0
                    self.tracks[i].KF.update([0],np.array([detections[assignment[i]][0]]),1,[0.1])
                    self.tracks[i].KF.update([1],np.array([detections[assignment[i]][1]]),1,[0.1])
                    self.tracks[i].prediction=self.tracks[i].KF.x;
                    #print("pre: ",self.tracks[i].prediction)
                    #self.tracks[i].prediction[1]=self.tracks[i].KF.x[1];
                else:
                    print(self.tracks[i].KF.x)
                    self.tracks[i].KF.update([0],np.array(np.array([0])),0,[0.1])
                    self.tracks[i].KF.update([1],np.array(np.array([0])),0,[0.1])
                self.tracks[i].prediction=self.tracks[i].KF.x
                    #self.tracks[i].prediction = self.tracks[i].KF.correct(
                     #                           np.array([[0], [0]]), 0)
                #print('det :',np.array([[detections[assignment[i]][0]],[detections[assignment[i]][1]]]))
                print('i : ', assignment[i], 'updated prediction : ',self.tracks[i].prediction)
                if(len(self.tracks[i].trace) > self.max_trace_length):
                    for j in range(len(self.tracks[i].trace) -
                                   self.max_trace_length):
                        del self.tracks[i].trace[j]
                
                
                if self.tracks[i].prediction.shape[0] == 2:
                    self.tracks[i].prediction=np.insert(self.tracks[i].prediction,self.tracks[i].prediction.shape[0],0)
                    self.tracks[i].prediction=np.insert(self.tracks[i].prediction,self.tracks[i].prediction.shape[0],0)
                    self.tracks[i].prediction=np.insert(self.tracks[i].prediction,self.tracks[i].prediction.shape[0],0)
                    self.tracks[i].prediction=np.insert(self.tracks[i].prediction,self.tracks[i].prediction.shape[0],0)
                
                self.tracks[i].trace.append(np.reshape(self.tracks[i].prediction,(6,1)))
                #print('cc : ',self.tracks[i].prediction.shape)
                self.tracks[i].KF.lastResult = self.tracks[i].prediction
                self.ls = assignment
        else:
            assignment = self.ls
            for i in range(len(assignment)):
                print('before only prediction: ', self.tracks[i].KF.x)
                self.tracks[i].KF.predict(0.08)
                print('after only prediction: ', self.tracks[i].KF.x)
                
                self.tracks[i].KF.update([0],np.array(np.array([0])),0,[0.3])
                self.tracks[i].KF.update([1],np.array(np.array([0])),0,[0.3])
                self.tracks[i].prediction=self.tracks[i].KF.x
                    #self.tracks[i].prediction = self.tracks[i].KF.correct(
                     #                           np.array([[0], [0]]), 0)
                #print('det :',np.array([[detections[assignment[i]][0]],[detections[assignment[i]][1]]]))
                print('i : ', assignment[i], 'updated prediction : ',self.tracks[i].prediction)
                if(len(self.tracks[i].trace) > self.max_trace_length):
                    for j in range(len(self.tracks[i].trace) -
                                   self.max_trace_length):
                        del self.tracks[i].trace[j]
                
                
                if self.tracks[i].prediction.shape[0] == 2:
                    self.tracks[i].prediction=np.insert(self.tracks[i].prediction,self.tracks[i].prediction.shape[0],0)
                    self.tracks[i].prediction=np.insert(self.tracks[i].prediction,self.tracks[i].prediction.shape[0],0)
                    self.tracks[i].prediction=np.insert(self.tracks[i].prediction,self.tracks[i].prediction.shape[0],0)
                    self.tracks[i].prediction=np.insert(self.tracks[i].prediction,self.tracks[i].prediction.shape[0],0)
                
                self.tracks[i].trace.append(np.reshape(self.tracks[i].prediction,(6,1)))
                #print('cc : ',self.tracks[i].prediction.shape)
                self.tracks[i].KF.lastResult = self.tracks[i].prediction

