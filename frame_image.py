# coding:utf-8
import cv2
import os

#这里只给出AnFpark里视频获取图片的代码，其它两个类似。
if __name__ == '__main__':
    rootdir = '/Users/gcf/Desktop/AnFpark'
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        vc = cv2.VideoCapture(path)
        rval, frame = vc.read()
        res = cv2.resize(frame, (352, 240), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('/Users/gcf/Desktop/image/' + 'AnFpark_' + str(i) + '.jpeg', res)
