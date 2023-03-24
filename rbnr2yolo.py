import scipy.io
import os, cv2
import numpy as np
import pybboxes as pbx

def bbox2yolo(a, size):
    b = (a[2], a[0], a[3], a[1])
    r = pbx.convert_bbox(b, from_type="voc", to_type="yolo", image_size=(size[1], size[0]))
    b = [0, r[0], r[1], r[2], r[3]]
    return b

folder = 'set1_org'
files = os.listdir(folder)

for file in files:
    if file.endswith("01.JPG.mat"):
        mat = scipy.io.loadmat(f'{folder}/{file}')
        bib = mat["tagp"]
        newBib = []

        for line in bib:
            nf = str(file)
            nf = nf.replace('.JPG.mat','')
            im = cv2.imread(f'{folder}/{nf}.JPG')
            size = im.shape
            line = bbox2yolo(line, size)
            if newBib == []:
                newBib = [line]
            else:
                newBib = np.concatenate([newBib,[line]])
            newBib = np.round(newBib, 4)
        
        f = open(f"{folder}/{nf}.txt", "w+")
        for line in newBib:
            line = str(line)
            line = line.replace('0.     ', '0 ')
            line = line.replace('  ', ' ')
            line = line.replace('   ', ' ')
            f.write(line[1:-1])
            f.write('\n')
        f.close()