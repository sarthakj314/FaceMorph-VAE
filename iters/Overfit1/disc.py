import os
import sys

urls = {'train1':'https://discordapp.com/api/webhooks/998729485421654036/DvzR66N8M9uUIJ2Dtwt8scuPu10IQIlebIlO9pTn9-1aAAoQWASS_yGIN9c2vsBnbuFV',
        'val1':'https://discordapp.com/api/webhooks/998646063172964403/NNH3R1zq4lFLaKjQKSLuM9AzMm43rwkPKbPBgwj64vFf8dSJNEcyRp4JOU4AaO2SAQ_U',
        'test1':'https://discordapp.com/api/webhooks/998729988603920485/pF2ljVj-d13H_El9i-2H1NUh7fhIAjHbccShekKlZQ5mPbn3JfLVMe6_FqF--SU3kJnd',
        'train2':'https://discord.com/api/webhooks/999221681564438568/y4dih5GMazTGqtqe66kmQJHzq01s_GwVhCuGIJXLbD-8FC6w2oOT9XcWa0xA971-MXJb',
        'val2':'https://discord.com/api/webhooks/999221904529432586/AL3sZpcHKqDLCntMVPBvptUcgKzRmCM_NLuHNAoya5r3lpqTXAF54pGZjxALRKoZibh2',
        'test2':'https://discord.com/api/webhooks/999222024952094810/qd--UnPC894f8FHd6H9mV2o6Ab_mINy4DZoonhNpqUEOUJ0dFKSlfVINJBy_ZiYl_YJF', 
        'train3':'https://discord.com/api/webhooks/1002559846471716904/uCpEotWeoMQAejJ4vxdUu4c0wof6kUagZwGMvPpyOnVehCp4iThqOFMbYnLMuXXhanA1',
        'val3':'https://discord.com/api/webhooks/1002559951547420732/5VBHNHMM-DptAQqJfoUmlOGa0qdEWfEFu7R8kfrR-6QBFXe5BTu79QoZewYt_cgRXmiN',
        'test3':'https://discord.com/api/webhooks/1002560052768550973/km64I1LMEex8KR_iog1p728V42HI8rkA1E2S5vXvkX0YBThU4P9E7kSxGIxhghKZLKH-',
        'train4':'https://discordapp.com/api/webhooks/1064676402449875034/ECXQ-Xq7y-7wSWTYcelrFPC_WcWO2RjxxgpaDM0LiN5_puyv9YVdoIhPM9bZRBbYOj9C'}

def disc_image(image, channel, user="Image-God"):
    url = urls[channel]
    command = 'curl -s -F \"file1=@'+image+'\" \"'+url+'\" > .trash'
    #print(command)
    os.system(command)

def disc_text(message, channel, user="Text-God"):
    url = urls[channel]
    command = 'curl -s -F \'payload_json={\"username\": \"'+user+'\", \"content\": \"'+message+'\"}\' \"'+url+'\" > .trash'
    #print(command)
    os.system(command)

#disc_image('/home/ubuntu/model/iters/efficientnet_v2_l-rerun/confusion_matrices/epoch0.png')
