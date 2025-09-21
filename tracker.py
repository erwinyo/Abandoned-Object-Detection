# DataFlair Abandoned object Tracker

import math

from ultralytics import YOLO
from collections import deque


class ObjectTracker:
    def __init__(self):
        self.model = YOLO("cls-model.pt")
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

        self.abandoned_temp = {}
        # self.abandoned_pred = {}

    def update(self, frame, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []
        abandoned_object = []
        # abandoned_pred = []
        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            x1, y1, x2, y2 = x, y, x + w, y + h
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                distance = math.hypot(cx - pt[0], cy - pt[1])

                if distance < 100:
                    # update the center point
                    self.center_points[id] = (cx, cy)

                    objects_bbs_ids.append([x, y, w, h, id, distance])
                    same_object_detected = True

                    #   Add same object to the abandoned_temp dictionary. if the object is
                    #   still in the temp dictionary for certain threshold count then
                    #   the object will be considered as abandoned object
                    if id in self.abandoned_temp:
                        if distance < 1:
                            if self.abandoned_temp[id] > 200:  # Threshold count
                                # cropped = frame[y1:y2, x1:x2]
                                # result = self.model.predict(
                                #     source=cropped, imgsz=256, conf=0.6, device=0
                                # )[0]
                                # probs = result.probs
                                # top1 = probs.top1
                                # top1_conf = probs.top1conf
                                # pred_object = self.model.names[top1]
                                # if top1_conf > 0.6:
                                #     abandoned_object.append([id, x, y, w, h, distance])
                                #     # abandoned_pred.append(pred_object)
                                #     self.abandoned_pred[id].append(pred_object)
                                abandoned_object.append([id, x, y, w, h, distance])
                            else:
                                # Increase count for the object
                                self.abandoned_temp[id] += 1
                    break

            # If new object is detected then assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                self.abandoned_temp[self.id_count] = 1
                # self.abandoned_pred[self.id_count] = deque(maxlen=10)
                objects_bbs_ids.append([x, y, w, h, self.id_count, None])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        abandoned_temp_2 = {}
        # abandoned_pred_2 = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, _ = obj_bb_id
            center = self.center_points[object_id]

            new_center_points[object_id] = center

            if object_id in self.abandoned_temp:
                counts = self.abandoned_temp[object_id]
                abandoned_temp_2[object_id] = counts

            # if object_id in self.abandoned_pred:
            #     preds = self.abandoned_pred[object_id]
            #     abandoned_pred_2[object_id] = preds

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        self.abandoned_temp = abandoned_temp_2.copy()
        # self.abandoned_pred = abandoned_pred_2.copy()
        # return objects_bbs_ids, abandoned_object, self.abandoned_pred
        return objects_bbs_ids, abandoned_object
