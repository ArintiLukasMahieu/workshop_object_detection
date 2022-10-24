# workshop_object_detection
Arinti DS workshop on object detection using background/foreground subtraction

## Problem statement
Given a video that starts with an empty labbench, detect and track the objects that are placed on the labbench without training any models.

## Description

In this repository you'll see an example python script that uses opencv's built-in background subtractor to try and achieve this.

The problems with the current approach are:
* The detections are very noisy; objects are detected that are not there.
* The detections are not stable; the same object is detected as different objects.
* The detections have limited memory; the object will lose its identity after a while.

Place the data for this exercise in the `data` folder and name it `video.mp4`.
