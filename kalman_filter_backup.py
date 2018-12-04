'''
    File name         : kalman_filter.py
    File Description  : Kalman Filter Algorithm Implementation
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np


class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self,centers):
        """Initialize variable used by Kalman Filter class
        Args:
            None
        Return:
            None
        """
        self.dt = 0.01  # delta time
        print('const : ',centers)
        self.A = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # matrix in observation equations
        self.u = np.array([[centers[0]],[centers[1]],[0.1],[0.1]])  # previous state vector
        #self.u = np.zeros((4, 1))  # previous state vector
        # (x,y) tracking object center
        self.b = np.array([[centers[0]], [centers[1]]])  # vector of observations
        #self.b = np.array([[0], [255]])  # vector of observations


        self.P = 0.3*np.diag((1.0, 1.0, 1.0, 1.0))  # covariance matrix
        self.F = np.array([[1.0, 0, self.dt, 0], [0.0, 1.0, 0, self.dt], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])  # state transition mat

        self.Q = 0.1*np.eye(self.u.shape[0])  # process noise matrix
        self.R = 0.3*np.eye(self.b.shape[0])  # observation noise matrix
        self.lastResult = np.array([[0], [255],[0],[0]])

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Args:
            None
        Return:
            vector of predicted state estimate
        """
        # Predicted state estimate
        self.u = np.round(np.dot(self.F, self.u))
        # Predicted estimate covariance
        self.P = np.add(np.dot(self.F, np.dot(self.P, self.F.T)),self.Q)
        self.lastResult = self.u  # same last predicted result
        return self.u

    def correct(self, b, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """

        if not flag:  # update using prediction
            self.b = np.array([self.lastResult[0],self.lastResult[1]])
        else:  # update using detection
            self.b = b
        #print('kkkkkkkkkkkkk : ',self.b.shape)
        C = np.add( np.dot(self.A, np.dot(self.P, self.A.T)) , self.R)
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.u = np.round(np.add(self.u, np.dot(K, np.subtract(self.b , np.dot(self.A,
                                                              self.u)))))
        self.P = np.subtract(self.P,np.dot(K, np.dot(self.A, self.P)))
        self.lastResult = self.u
        return self.u
