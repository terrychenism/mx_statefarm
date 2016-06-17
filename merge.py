from collections import defaultdict
from operator import add
cv_list = [1,2,3,4,5]
epoch = 14
pred_dict = dict()
for cv in cv_list:
    fi = open('submission_test_batch_%d_epoch_%d.csv'%(cv, epoch),'r')
    header = fi.readline()
    for f in fi:
        ll = f.strip().split(',')
        img = ll[0]
        pred = map(float, ll[1:])
        if not img in pred_dict:
            pred_dict[img]=pred
        else:
            old_pred = pred_dict[img]
            pred_dict[img] = map(add, old_pred, pred)
    fi.close()

#expore the mean value of predicted probability
fw_out = open('submission_epoch_%d_cv.csv'%(epoch),'w')
fw_out.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
for img, pred in sorted(pred_dict.iteritems()):
    pred = ['%.6f'%(p/len(cv_list)) for p in pred]
    fw_out.write('%s,%s\n'%(img,','.join(pred)))

fw_out.close()
