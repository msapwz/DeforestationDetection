from sklearn.model_selection import train_test_split
import os
import shutil
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('path',type=str,help='path with the outputs from step4 (it ends with the size of the chips: "512","256"...)')
parser.add_argument('outpath',type=str,help="Place to store the output.")
args = parser.parse_args()

def createpath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def prepare_outpath(path,outpath):
    name = os.path.basename(os.path.dirname(path))
    outpath = os.path.join(outpath,name,os.path.basename(path))

    createpath(os.path.join(outpath,"A"))
    createpath(os.path.join(outpath,"B"))
    createpath(os.path.join(outpath,"label"))
    createpath(os.path.join(outpath,"list"))

    return outpath


path = args.path
outpath = args.outpath
# path = 'dataset/chips/NDVI/256'
# outpath = 'dataset/chipsCF2'
outpath = prepare_outpath(path,outpath)

path1 = os.path.join(path,'inputs1')
X = list(set([i[0:11] for i in os.listdir(path1)]))

# X = [i for i in os.listdir(path1) if i.find('rot=')==-1]

X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
X_train, X_val = train_test_split(X_train,test_size=0.25, random_state=1)

X_train = [i for i in os.listdir(path1) if i[0:11] in X_train]
X_test = [i for i in os.listdir(path1) if i[0:11] in X_test]
X_val = [i for i in os.listdir(path1) if i[0:11] in X_val]

ltrain = len(X_train)
l_test = len(X_test)
l_val = len(X_val)
N = sum((l_test,ltrain,l_val))
pct_train = 100*ltrain/N
pct_test = 100*l_test/N
pct_val = 100*l_val/N

print(f"Pct for training {round(pct_train,2)}%, number of files = {ltrain}")
print(f"Pct for test {round(pct_test,2)}%, number of files = {l_test}")
print(f"Pct for validation {round(pct_val,2)}%, number of files = {l_val}")

def save_files(files,path,outpath,name,rot=[90,180]):
    # files_rotated = []
    # for r in rot:
    #     files_rotated.append([os.path.splitext(i)[0]+f'_rot={r}'+os.path.splitext(i)[-1] for i in files])
    # for f in files_rotated:
    #     files+=f
    path1 = os.path.join(path,'inputs1')
    path2 = os.path.join(path,'inputs2')
    label = os.path.join(path,'labels')

    outA = os.path.join(outpath,"A")
    outB = os.path.join(outpath,"B")
    outL = os.path.join(outpath,"label")
    outl = os.path.join(outpath,"list")
    with open(os.path.join(outl,name+".txt"),mode='w') as f:
        for file in tqdm(files):
            srcA = os.path.join(path1,file)
            srcB = os.path.join(path2,file)
            srcL = os.path.join(label,file)

            dstA = os.path.join(outA,f'{name}_{file}')
            dstB = os.path.join(outB,f'{name}_{file}')
            dstL = os.path.join(outL,f'{name}_{file}')

            shutil.copyfile(srcA,dstA)
            shutil.copyfile(srcB,dstB)
            shutil.copyfile(srcL,dstL)

            f.write(f'{name}_{file}\n')


save_files(X_train,path,outpath,'train')
save_files(X_test,path,outpath,'test')
save_files(X_val,path,outpath,'val')