#!/usr/bin/env python

import os
import time

import cv2


class Tracker(object):
    def __init__(self):
        self.trakcer, self.tracker_name = self._select_tracker()

        if os.path.exists('./videos/') is False:
            os.makedirs('./videos')

        self.cap = cv2.VideoCapture(0)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = tuple((w, h))
        frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(
            './videos/{}.mp4'.format(self.tracker_name),
            fourcc, frame_rate, size)

    def _select_tracker(self):
        print('Which Tracker API do you use??')
        print('0: Boosting')
        print('1: MIL')
        print('2: KCF')
        print('3: MLD')
        print('4: MedianFlow')
        choice = int(input('Please select your tracker number: '))

        if choice == 0:
            tracker = cv2.TrackerBoosting_create()
        elif choice == 1:
            tracker = cv2.TrackerMIL_create()
        elif choice == 2:
            tracker = cv2.TrackerKCF_create()
        elif choice == 3:
            tracker = cv2.TrackerTLD_create()
        elif choice == 4:
            tracker = cv2.TrackerMedianFlow_create()
        else:
            raise ValueError('Tracker is undified!!')
            return

        self.tracker = tracker
        self.tracker_name = str(tracker).split()[0][1:]

        return self.tracker, self.tracker_name

    def _iterate(self):
        while True:
            ret, frame = self.cap.read()
            success, roi = self.tracker.update(frame)

            (x, y, w, h) = tuple(map(int, roi))

            if success:
                p1 = (x, y)
                p2 = (x + w, y + h)
                cv2.rectangle(frame, p1, p2, color=(0, 255, 0), thickness=3)
            else:
                cv2.putText(frame, text='Tacking failed!!',
                            org=(500, 400),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=3)

            cv2.putText(frame, text=self.tracker_name,
                        org=(20, 600),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=3)

            self.writer.write(frame)
            cv2.imshow(self.tracker_name, frame)
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break

    def start(self):
        time.sleep(1)

        ret, frame = self.cap.read()
        roi = cv2.selectROI(frame, False)
        self.tracker.init(frame, roi)

        self._iterate()

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()


def main():
    tracker = Tracker()
    tracker.start()


if __name__ == '__main__':
    main()
