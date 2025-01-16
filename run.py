'''''''''''''''
How to run the model:

python3 run.py testing_mode -> testing mode

python3 run.py training_mode -n [name] -d [description] -> training mode

python3 run.py resume -> will resume from last run

testing mode -> Will not record any data, strictly for testing if code will run

training mode -> Will record:
    - Time of creation
    - Python file of model
    - .pth file of best model
    - Nueral network class file
    - Description included in meta.txt
'''''''''''''''


import os
import sys
import datetime

#user = '/'.join(os.getcwd().split('/')[:3])
cwd = os.getcwd()
args = sys.argv
os.chdir(cwd)

train = (args[1] == 'training_mode')

# datetime.datetime.now() to get time/date

if len(args) == 1:
    print("Please include \'testing_mode\' or \'training_mode\' for use")
    sys.exit()

if args[1]=='resume':
    if len(args)>2: last = args[2]
    else: last = open('.last-run').read().split('/')[-1]
    print('Resuming '+last)
    os.chdir('iters/'+last)
    os.system('python3 model.py /iters/'+last+' train')
    sys.exit()

print('You are currently in', args[1], 'mode')

cnt=0
models = {}
for i in os.listdir('models'):
    if 'model' in i:
        cnt += 1
        models[cnt] = i
        print(cnt,':',i)
print()
if len(args)==1:
    print('Which model would you like to test?')
else:
    print('Which model would you like to train?')
c1 = int(input())
model = 'models/'+models[c1]


if train:
    if '-n' in args:
        begin = args.index('-n')+1
        if '-d' in args:
            end = args.index('-d')
            name = ' '.join(args[begin:end])
        else:
            name = ' '.join(args[begin:])
    else:
        name = 'iter%s' % (len(os.listdir('iters')))
    name = name

    if '-d' in args:
        begin = args.index('-d')+1
        desc = ' '.join(args[begin:])
    else:
        desc = ''

    with open('.last-run', 'w') as f:
        f.write('/iters/'+name)
else:
    name = 'testing_mode'

os.system('mkdir iters/'+name)
os.chdir('iters/'+name)

os.system('cp '+cwd+'/'+model+' model.py')
os.system('cp '+cwd+'/disc.py disc.py')
os.system('cp -r '+cwd+'/models/networks networks')
os.system('mkdir outputs')

if train:
    with open('.last-epoch', 'w') as f:
        f.write('0')
    with open('meta.txt', 'w') as f:
        date = str(datetime.datetime.now())
        f.writelines([model, '\n', date, '\n', desc])
    os.system('python3 model.py /iters/'+name+' train')
else:
    os.system('python3 model.py /iters/'+name+' test')
