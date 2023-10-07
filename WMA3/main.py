import cv2
import numpy as np

def resize(img, s):
    image = img.copy()
    h = int(image.shape[0] / s)
    w = int(image.shape[1] / s)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return image

def finderSIFT(sample, video):
    
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('resultSIFT.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

    gimg1 = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(gimg1, None)

    while(video.isOpened()):
        recieved, frame = video.read()
        if not recieved:
            break   

        gimg2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints_2, descriptors_2 = sift.detectAndCompute(gimg2, None)
        
        matches = matcher.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x:x.distance)

        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = sample.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        frame=cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        # key = cv2.waitKey(2)
        # if key == ord('q'):
        #     break
        
        # cv2.imshow('Video', frame)
        result.write(frame)
    video.release()
    result.release()
    cv2.destroyAllWindows()

def finderORB(sample, video):
    
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('resultORB.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

    gimg1 = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    orb = cv2.ORB_create()
    keypoints_1, descriptors_1 = orb.detectAndCompute(gimg1, None)

    while(video.isOpened()):
        recieved, frame = video.read()
        if not recieved:
            break   

        gimg2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints_2, descriptors_2 = orb.detectAndCompute(gimg2, None)
        
        matches = matcher.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x:x.distance)

        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = sample.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        frame=cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        # key = cv2.waitKey(2)
        # if key == ord('q'):
        #     break
        
        # cv2.imshow('Video', frame)
        result.write(frame)
    video.release()
    result.release()
    cv2.destroyAllWindows()

def main():
    sample = cv2.imread('photo_2_query.jpg')
    sample = resize(sample, 4)   
    cap = cv2.VideoCapture('video_3_query.mp4')
    train = cv2.imread('photo_2_train.jpg');
    #finderSIFT(sample, cap)
    finderORB(sample, train)


if __name__=="__main__":
    main()

