#!/usr/bin/env python

import os

import cv2
import numpy as np


class OpticalFlow(object):
    def __init__(self):
        if os.path.exists('./videos/opticalflow') is False:
            os.makedirs('./vodeos/opticalflow')

        self.file_name = str(input('Enter saving file name: '))

        self.cap = cv2.VideoCapture(0)

        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            crteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.color = np.random.randint(0, 255, (100, 3))

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = tuple((w, h))
        frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(
            './videos/opticalflow/{}.mp4'.format(
                self.file_name), fourcc, frame_rate, size
        )

    def _iterate(self, old_gray, p0, mask):
        while True:
            ret, frame = self.cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **self.lk_params)

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(img=mask, pt1=(a, b), pt2=(c, d),
                                color=self.color[i].tolist(), thickness=2)

                frame = cv2.circle(img=frame,
                                   center=(a, b),
                                   radius=5,
                                   color=self.color[i].tolist(),
                                   thickness=-1)

            img = cv2.add(frame, mask)

            self.writer.write(img)
            cv2.imshow('frame', img)
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    def start(self):
        ret, old_frame = self.cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(
            old_gray, mask=None, **self.feature_params)

        mask = np.zeros_like(old_frame)

        self._iterate(old_gray, p0, mask)

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()


def main():
    flow = OpticalFlow()
    flow.start()


if __name__ == '__main__':
    main()
