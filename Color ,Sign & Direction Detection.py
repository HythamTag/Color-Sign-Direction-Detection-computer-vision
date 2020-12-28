# 72 max max 43 46 97

# import the necessary packages


import cv2
import numpy as np
import math

    cap = cv2.VideoCapture(0)

h=640
w=640



w=640
h=640


minDiff = 1000000000000
minSquareArea = 5000
match = -1


ReferenceImages = ["tech.jpg", "super.jpg", "noth.jpg"]
ReferenceTitles = ["T", "S", "n" ]

class Symbol:
    def __init__(self):
        self.img = 0
        self.name = 0


#define class instances (3 objects for 3 different images)
symbol= [Symbol() for i in range(3)]
symbol_binary= [Symbol() for i in range(3)]


def readRefImages():
    for count in range(3):
        imagee = cv2.imread(ReferenceImages[count], cv2.COLOR_BGR2GRAY)
        symbol[count].img = cv2.resize(imagee,(w//2,h//2),interpolation = cv2.INTER_AREA)
        symbol[count].name = ReferenceTitles[count]
        symbol_binary[count].img = Resize_hreshold_warped(symbol[count].img , 0,1,2)

        #cv2.imshow('tttt',symbol_gray[1].img);











def Resize_hreshold_warped(image, do_resize, do_thresh,thresh):
    # Resize the corrected image to proper size & convert it to grayscale

    if do_resize == 1:
        image = cv2.resize(image, (w, h))
    if do_thresh == 1:
        warped_new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Smoothing Out Image
        blur = cv2.GaussianBlur(warped_new_gray, (5, 5), 0)

        # Calculate the maximum pixel and minimum pixel value & compute threshold
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
        threshold = (min_val + max_val) // thresh
        # Threshold the image
        ret, warped_processed = cv2.threshold(warped_new_gray, threshold, 255, cv2.THRESH_BINARY)

        # return the thresholded image
        return warped_processed
    else:
        return image


def Color_Detect(image):
    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    red_max_area =0
    blue_max_area =0

    upper_red = np.array([180, 256, 195])
    lower_red = np.array([153, 119, 0])

    upper_blue = np.array([118, 256, 195])
    lower_blue = np.array([102, 175, 15])

    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blur_red = cv2.medianBlur(mask_red.copy(), 15)
    dilation_red = cv2.dilate(mask_blur_red, kernel, iterations=1)
    # dilation_red = cv2.medianBlur(mask_red.copy(), 31)
    process_red, contours_red, hierarchy = cv2.findContours(dilation_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blur_blue = cv2.medianBlur(mask_blue.copy(), 15)
    dilation_blue = cv2.dilate(mask_blur_blue, kernel, iterations=1)
    # dilation_blue = cv2.medianBlur(mask_blue.copy(), 15)
    process_blue, contours_blue, hierarchy = cv2.findContours(dilation_blue.copy(), cv2.RETR_TREE,
                                                              cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_red) != 0:
        red_max_area = max(contours_red, key=cv2.contourArea)
        a, b, c, d = cv2.boundingRect(red_max_area)


    if len(contours_blue) != 0:
        blue_max_area = max(contours_blue, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(blue_max_area)


    if len(contours_red) != 0 and len(contours_blue) !=0:
        if cv2.contourArea(red_max_area) > cv2.contourArea(blue_max_area):
            color = 'Red'
            process = process_red
            cv2.rectangle(image, (a, b), (a + c, b + d), (0, 255, 255), 3)
            cv2.putText(image, "Red", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow('img', image)
            cv2.imshow('process', process_red)
            key = cv2.waitKey(1) & 0xFF
            return color, image, process

        elif cv2.contourArea(blue_max_area) > cv2.contourArea(red_max_area):
            color = 'Blu'
            process = process_blue
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)
            cv2.putText(image, "Blue", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow('img', image)
            cv2.imshow('process', process_blue)
            key = cv2.waitKey(1) & 0xFF
            return color, image, process
    elif len(contours_red) != 0 and len(contours_blue) ==0:
        color = 'Red'
        process = process_red
        cv2.rectangle(image, (a, b), (a + c, b + d), (0, 255, 255), 3)
        cv2.putText(image, "Red", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow('img', image)
        cv2.imshow('process', process_red)
        key = cv2.waitKey(1) & 0xFF
        return color, image, process

    elif len(contours_red) == 0 and len(contours_blue) !=0:
        color = 'Blu'
        process = process_blue
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(image, "Blue", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow('img', image)
        cv2.imshow('process', process_blue)
        key = cv2.waitKey(1) & 0xFF
        return color, image, process
    else:
        color = 'n'
        return color, image,np.zeros(image.shape, dtype=np.uint8)

    cv2.imshow('img', image)
    key = cv2.waitKey(1) & 0xFF
def Detect_Direction(camera, threshold):
        CountL = 0
        CountR = 0

        Direction = "n"

        org = camera.copy()
        process = Resize_hreshold_warped(camera, 0, 1, threshold)

        process, contours, hierarchy = cv2.findContours(process, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # process = cv2.GaussianBlur(process, (3, 3), 0);
        process = cv2.medianBlur(process, 5)
        process = cv2.medianBlur(process, 3)
        process = cv2.medianBlur(process, 3)
        for i in range(len(contours)):
            cnt = contours[i]

            if 1000 < cv2.contourArea(cnt) < 50000:
                epsilon = 0.009 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) == 7:

                    # cv2.drawContours(org, [approx], 0, (0, 0, 255), 2)
                    hull = cv2.convexHull(cnt)

                    cnt_hull_ratio = round(cv2.contourArea(cnt) / cv2.contourArea(hull), 1)

                    # if cnt_hull_ratio not in {0.7, 0.8, 0.9}:
                    # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


                    if cnt_hull_ratio in {0.7, 0.8, 0.9}:

                        # cv2.drawContours(org, [cnt], 0, (0, 0, 255), 2)
                        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
                        extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
                        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
                        extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
                        # cv2.circle(org ,extLeft,5,(255,3,200),-1)
                        # cv2.circle(org, extRight, 5, (255, 3, 200), -1)



                        hull = cv2.convexHull(cnt, returnPoints=False)
                        defects = cv2.convexityDefects(cnt, hull)

                        count_defects = 0
                        max_d = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            if d > max_d:
                                max_d = d
                                farthest = far
                                # print(d)
                                # cv2.line(org, start, end, [0, 255, 0], 2)


                                # print(((int(extTop[1]) - int(extBot[1]))*20))
                                # cv2.circle(org, farthest, 5, [0, 0, 255], -1)

                        try:

                            Horz_ratio1_R = int(farthest[0]) - int(extLeft[0]) > int(extRight[0]) - int(
                                farthest[0]) and round((int(farthest[0]) - int(extLeft[0])) / (
                                int(extRight[0]) - int(farthest[0]))) in (3, 4, 5)

                            Horz_ratio1_L = int(farthest[0]) - int(extLeft[0]) < int(extRight[0]) - int(
                                farthest[0]) and round((int(extRight[0]) - int(farthest[0])) / (
                                int(farthest[0]) - int(extLeft[0]))) in (3, 4, 5)

                            Ver_Horz_ratio = round(int(extBot[1] - extTop[1]) / int(extRight[0] - extLeft[0]), 1) in {
                            0.4, 0.5, 0.6, 0.7}

                            # print(int(extBot[1] - extTop[1]) / int(extRight[0] - extLeft[0]))

                            # if (Horz_ratio1_R !=1 and Ver_Horz_ratio !=1) or (Horz_ratio1_L!=1 and Ver_Horz_ratio !=1 ):

                            #    print(Horz_ratio1_R, Horz_ratio1_L , Ver_Horz_ratio)
                            #    print(int(extBot[1] - extTop[1]) / int(extRight[0] - extLeft[0]))
                            # print('end')



                        except Exception:
                            # print('Zero Div')
                            pass
                        else:
                            if Horz_ratio1_R and Ver_Horz_ratio:
                                cv2.drawContours(org, [approx], 0, (0, 0, 255), 2)
                                cv2.putText(org, "Right", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                                            cv2.LINE_AA)

                                Direction = "R"
                                cv2.imshow('process', process)
                                cv2.imshow('img', org)
                                # print('Right')
                                # print(round(int(extBot[1] - extTop[1]) / int(extRight[0] - extLeft[0]), 1))

                            elif Horz_ratio1_L and Ver_Horz_ratio:
                                cv2.drawContours(org, [approx], 0, (0, 0, 255), 2)
                                cv2.putText(org, "Left", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                                            cv2.LINE_AA)
                                Direction = "L"
                                cv2.imshow('process', process)
                                cv2.imshow('img', org)
                                # print('Left')
                                # print(round(int(extBot[1] - extTop[1]) / int(extRight[0] - extLeft[0]), 1))

        return Direction, org, process

        # print(cv2.contourArea(contours[1]))

        # print(org.shape)
def Sign_Detect(image ,threshold):
    match = -1
    output = image.copy()
    out = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # gray = cv2.medianBlur(gray, 3)
    gray = cv2.medianBlur(gray, 15)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=65, minRadius=70, maxRadius=0)
    thresh_ORG = Resize_hreshold_warped(output, 0, 1, threshold)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:

            if x - r - 1 >= 1 and x + r + 1 <= 639 and y - r - 1 >= 1 and y + r + 1 <= 639:

                thresh_ORG = Resize_hreshold_warped(output ,0 ,1, threshold)
                thresh_ROG = thresh_ORG[(y - r):(y + r), (x - r):(x + r)]

                if thresh_ORG[y, x - r + 1] == 0 and thresh_ORG[y, x + r - 1] == 0 and thresh_ORG[y, x - r - 1] == 255 and thresh_ORG[y, x + r + 1] == 255:
                    #print(r)

                    Last = cv2.resize(thresh_ROG, (w // 2, h // 2))

                    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    #cv2.putText(output, str(x) + ' ' + str(y), (x - 70, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2, cv2.LINE_AA)


                    minDiff = 1000000000000
                    for i in range(3):

                        diffImg = cv2.bitwise_xor(Last, symbol_binary[i].img)
                        white = cv2.countNonZero(diffImg);
                        height, width = diffImg.shape[:2]
                        size = width * height
                        black = size - white
                        # print(symbol[i].name, diff)

                        if white < minDiff:
                            match = i
                            minDiff = white

                    #diffImg = cv2.bitwise_xor(Last, symbol_binary[match].img)

                    #print(symbol[match].name, size, white, black)

                    #cv2.putText(out, symbol[match].name, (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
                    cv2.putText(output, symbol[match].name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    #print(symbol[match].name)
                    return symbol[match].name, output, thresh_ORG






    return 'n', output, thresh_ORG




def Sign_Detec_Rep(threshold,Repeat_Times):

    CountT = 0

    readRefImages()
    for i in range(0, Repeat_Times):

        ret, cam = cap.read()
        Sign, org, process = Sign_Detect(cam,threshold)
        #print(Sign)

        cv2.imshow('process', process)
        cv2.imshow('img', org)
        key = cv2.waitKey(1) & 0xFF

        if Sign == "T":
            CountT += 1

    print(CountT)
    if CountT > Repeat_Times // 5 :
        Sign = 'T'
        return Sign
    else:
        Sign = 'n'
        return Sign
def Detect_Dirc_Rep(threshold, Repeat_Times):
    CountR = 0
    CountL = 0

    for i in range(0, Repeat_Times):

        ret, cam = cap.read()
        Direction, org, process = Detect_Direction(cam, threshold)

        cv2.imshow('process', process)
        cv2.imshow('img', org)
        key = cv2.waitKey(1) & 0xFF

        if Direction == "R":
            CountR += 1
            CountL = 0
        elif Direction == "L":
            CountL += 1
            CountR = 0

    if CountR > Repeat_Times // 2 and CountR > CountL:
        Direction = 'R'
        return Direction
    elif CountL > Repeat_Times // 2 and CountL > CountR:
        Direction = 'L'
        return Direction
    else:
        Direction = 'n'
        return Direction
def Color_Detect_Rep(Repeat_Times):
    Count_Red = 0
    Count_Blue = 0

    for i in range(0, Repeat_Times):

        ret, cam = cap.read()
        Color, org, process = Color_Detect(cam)

        cv2.imshow('process', process)
        cv2.imshow('img', org)
        key = cv2.waitKey(1) & 0xFF

        if Color == "Red":
            Count_Red += 1
            Count_Blue = 0
        elif Color == "Blu":
            Count_Blue += 1
            Count_Red = 0

    if Count_Red >= Repeat_Times // 2 and Count_Red >= Count_Blue:
        Color = 'Red'
        return Color
    elif Count_Blue >= Repeat_Times // 2 and Count_Blue >= Count_Red:
        Color = 'Blue'
        return Color
    else:
        Color = 'n'
        return Color










while (1):


    color =Color_Detect_Rep(100)
    print(color)

    #direction = Detect_Dirc_Rep(2, 100)
    #print(direction)

    #sign = Sign_Detec_Rep(2, 100)
    #print(sign)










cv2.destroyAllWindows()


