from scipy.interpolate import CubicSpline
import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
import serial
import re
from datetime import datetime

start = time.time()

cam = cv.VideoCapture(1)
# cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
# cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

if cam.isOpened():
    rval, frame = cam.read()
else:
    rval = False

now = datetime.now()

name = now.strftime("cv_results/CV_" + "%d_%m_%H_%M_%S.dat")
# print(name)
fid = open(name, 'w')
fid.flush()
fid.write(now.strftime("%d_%m_%H_%M_%S.png\n"))

name2 = now.strftime("fsr_results/FSRs_" + "%d_%m_%H_%M_%S.dat")
# print(name2)
fsrs = open(name2, 'w')
fsrs.flush()
fsrs.write(now.strftime("graphs/graph_" + "%d_%m_%H_%M_%S.png\n"))

axe = []
aye = []
bxe = []
bye = []
cxe = []
cye = []
dxe = []
dye = []

list_time = []

ser_a = []
ser_b = []
ser_c = []
ser_d = []
ser_e = []
ser_f = []
ser_g = []
ser_readings = [ser_a, ser_b, ser_c, ser_d, ser_e, ser_f, ser_g]
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']

ib = 0
ic = 0
ig = 0

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False)
top, middle, bottom = axes.flatten()

# Before entering the loop
curve_line, = bottom.plot([], [], lw=2)
scatter_points = bottom.scatter([], [], c='r', marker='o', zorder=3)

def update_curve(x_plot, y_plot):
    curve_line.set_data(x_plot, y_plot)
    bottom.relim()
    bottom.autoscale_view()

def update_scatter(x_array, y_array):
    scatter_points.set_offsets(np.c_[x_array, y_array])

def redraw_figure():
    plt.draw()
    plt.pause(0.00001)

with serial.Serial('COM7', 9600) as ser:
    for ie in range(2):
        for ib in range(7):
            temp = ''.join(re.findall("\d", str(ser.readline())))
            ser_readings[ib].append(temp)
            # print('FSR Reading ' + letters[ib] + ': ' + str(ser_readings[ib]))

        for ic in range(7):
            ser_readings[ic].pop()
            # print('FSR Reading ' + letters[ic] + ': ' + str(ser_readings[ic]))

    while_start_time = time.time()

    while rval:
        start_loop = time.time()
        rval, frame = cam.read()
        t = time.time() - start
        # frame = cv.resize(frame, (1080,720), interpolation=cv.INTER_AREA)

        # Grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('Grayscale', gray)

        # Threshold the image to get the white shapes
        _, thresh = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)
        cv.imshow('Thresh', thresh)

        # Find contours in the thresholded image
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Draw contours
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)

        # Define variables
        # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        color = (50,50,50)

        for ia in range(len(contours)):
            mu = cv.moments(contours[ia])
            mc = [None] * len(contours)
            # add 1e-5 to avoid division by zero
            if mu['m00'] > 1500:
                mc = (mu['m10'] / (mu['m00'] + 1e-5), mu['m01'] / (mu['m00'] + 1e-5))
                cv.drawContours(drawing, contours, ia, color, 2)
                cv.circle(drawing, (int(mc[0]), int(mc[1])), 4, color, -1)

        # Initialize lists to store the x and y coordinates of the centers of the white shapes
        x_coords = []
        y_coords = []

        # Loop over the contours and find the centers of the white shapes
        for contour in contours:
            area = cv.contourArea(contour)

            if 500 < area < 6000:  # Filter out small contours to reduce noise
                # Fit a circle to the contour
                (x, y), radius = cv.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                # Check if the circle is roughly circular (aspect ratio close to 1)
                aspect_ratio = cv.fitEllipse(contour)[1][0] / cv.fitEllipse(contour)[1][1]

                if 0.8 < aspect_ratio < 1.2:
                    # Draw the circle on the original image
                    cv.circle(frame, center, radius, (0, 0, 255), 2)
                    cv.circle(drawing, center, radius, (0, 0, 255), 2)
                    fid.write(str(int(x)) + ' ' + str(int(y)) + ' ')

                    # Store the x and y coordinates of the center of the white shape
                    x_coords.append(int(x))
                    y_coords.append(int(y))
        try:
            axe.append(x_coords[0])
            aye.append(y_coords[0])
            bxe.append(x_coords[1])
            bye.append(y_coords[1])
            cxe.append(x_coords[2])
            cye.append(y_coords[2])
            dxe.append(x_coords[3])
            dye.append(y_coords[3])

        # Print the x and y coordinates of the centers of the white shapes
        # print("x coordinates:", x_coords)
        # print("y coordinates:", y_coords)
        # print(axe)
        # print(aye)
        # print(bxe)
        # print(bye)
        # print(cxe)
        # print(cye)
        # print(dxe)
        # print(dye)

            cv.imshow('Contours', drawing)
            cv.imshow("Image with white shapes", frame)

            current_time = time.time()
            list_time.append(current_time-while_start_time)
            # print(list_time)

            fid.write("\n")
            top.plot(list_time, aye, c='b')
            top.plot(list_time, bye, c='k')
            top.plot(list_time, cye, c='r')
            top.plot(list_time, dye, c='g')

            # try:
            id = 0
            for id in range(7):
                temp = int(''.join(re.findall("\d", str(ser.readline()))))
                ser_readings[id].append(temp)
                print('FSR Reading ' + letters[id] + ': ' + str(ser_readings[id]))
                fsrs.write(str(temp) + ' ')

        except IndexError:
            print("Unable to read on of the FSRs")

        fsrs.write("\n")

        for ih in range(len(ser_readings)):
            middle.plot(list_time, ser_readings[ih], c=colors[ih])

        x_array = np.array(x_coords)
        y_array = np.array(y_coords)

        # Sort the data points by their x values to ensure a proper interpolation
        sorted_indices = np.argsort(x_array)
        x_array = x_array[sorted_indices]
        y_array = y_array[sorted_indices]

        # Fit a cubic spline to the data points
        spline = CubicSpline(x_array, y_array)

        # Generate the x and y values for the curve
        x_plot = np.linspace(x_array.min(), x_array.max(), 100)
        y_plot = spline(x_plot)

        # Update the curve
        update_curve(x_plot, y_plot)
        update_scatter(x_array, y_array)
        redraw_figure()

        # time.sleep(0.1)

        # print(str(time.time() - start_loop))

        ig += 1

        # Exit
        key = cv.waitKey(1)
        if key == 27:  # Exit on ESC
            fid.close()
            fsrs.close()
            fig.savefig(now.strftime("graphs/graph_" + "%d_%m_%H_%M_%S.png"))
            break
        # except:
        #     print("Move your hand")
