wgs = [0.08]
ts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1,1.2,1.5,2,3,5,10]
mgs = [0.7]
lr = [1.7]

for wg in wgs:
    for l in lr:
        for t in ts:
            for m in mgs:
                tmp = 'CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/new_env/bin/python3.6 main.py --lr '\
                      +str(l)+' --weight_ture '+str(wg)+ ' --threshold '+str(t) + ' --margin ' +str(m) +\
                      ' --run 1211 ' +'--ep 150 ' + ' --loss THLI '
                print(tmp)