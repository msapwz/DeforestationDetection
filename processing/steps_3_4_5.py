import os,sys
from tqdm import tqdm
import argparse

parser =  argparse.ArgumentParser(description="Run steps 3,4 and 5 in a loop")
parser.add_argument("file",type=str,help="Text file with all the files that will be used to create the chips")
parser.add_argument("-r",'--resolution',type=int,default=512,help="Size of the chips")
parser.add_argument("-c",'--compositions',type=str,default='/dataset/area_89/compos',help="Folder where the compositions are stored")
parser.add_argument("-l",'--labels',type=str,default='/dataset/area_89/labels',help="Folder where the labels will be stored")
parser.add_argument("-x",'--chips',type=str,default='/dataset/area_89/chips',help="Folder where the chips will be stored")
parser.add_argument("-o",'--output',type=str,default='/dataset/area_89/chips_CF',help="Folder where the final files will be stored")
parser.add_argument("-y",'--yearly',type=str,default='/dataset/yearly_deforestation_biome.shp',help="yearly deforestation file location")
parser.add_argument("-f",'--format_file',type=str,default='png',help="Format of the chips")
parser.add_argument("-p","--pct_class1",type=float,default=0,help="Minimum percentage of class 1.")
args = parser.parse_args()

file = args.file
res = args.resolution
compos_folder = args.compositions
labels_folder = args.labels
chips_folder = args.chips
outpath = args.output
df = args.yearly
format_file = args.format_file
pct_class1 = args.pct_class1

num_lines = sum(1 for line in open(file,'r'))


# TODO: Change here if needed, the location of the steps 3, 4 and 5.
s1 = '/dataset/deforestation-project/step3_create_labels.py'
s2 = '/dataset/deforestation-project/step4_create_chips.py'
s3 = '/dataset/deforestation-project/step5_chips_ChangeFormer.py'
# s1 = 'step3_create_labels.py'
# s2 = 'step4_create_chips.py'
# s3 = 'step5_chips_ChangeFormer.py'

with open(file) as f:
    for i in tqdm(f, total=num_lines):
        (file1,file2) = i.replace('\n','').split(',')

        #print(file1,file2)
        orb_tile = file1[33:44]
        date1 = file1[11:19]
        date2 = file2[11:19]

        file1 = os.path.join(compos_folder,orb_tile,file1)
        file2 = os.path.join(compos_folder,orb_tile,file2)

        if os.path.exists(file1) and os.path.exists(file2):
            
            command = f'python3 {s1} {df} {file1} {file2} {labels_folder}'
            #print(command)
            os.system(command)

            # s2 = '/dataset/deforestation-project/step4_create_chips.py'
            # # s2 = 'step4_create_chips.py'
            name = f'{orb_tile}_{date1}_{date2}'
            path = f'{labels_folder}/{name}'

            command = f'python3 {s2} {path} {chips_folder} -s {res} -n 0.99 -f {format_file} -p {pct_class1}'
            os.system(command)

        else:
            print(f"Files {file1} and {file2} do not exists")

#############################################################################
s3 = '/dataset/deforestation-project/step5_chips_ChangeFormer.py'
# # s3 = 'step5_chips_ChangeFormer.py'
# path = f'{chips_folder}/{name}/{res}'
path = f'{chips_folder}/{res}'
command = f'python3 {s3} {path} {outpath}'
print("STEP 5:")
os.system(command)
#############################################################################


print("DONE ALL")