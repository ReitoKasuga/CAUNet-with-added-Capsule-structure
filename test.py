import torch
import torch.nn as nn
from metric import get_stoi, get_pesq
from checkpoints import Checkpoint
from torch.utils.data import DataLoader
from utils import snr, numParams
from eval_composite import eval_composite
from AudioData import EvalDataset, EvalCollate
from network import Net
import h5py
import os
from scipy.io.wavfile import write
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sr = 16000
test_file_list_path_noisy = 'S:/Nagamori/Wave-U-Net-for-Speech-Enhancement-NNabla-master/data/noisy_testset_wav'
test_file_list_path_clean = 'S:/Nagamori/Wave-U-Net-for-Speech-Enhancement-NNabla-master/data/clean_testset_wav'

# test function
def evaluate(net, eval_loader):
    net.eval()

    print('********Starting metrics evaluation on test dataset**********')
    total_stoi = 0.0
    total_ssnr = 0.0
    total_pesq = 0.0
    total_csig = 0.0
    total_cbak = 0.0
    total_covl = 0.0

    with torch.no_grad():
        count, total_eval_loss = 0, 0.0
        f = open("D:/Desktop/kasuga/研究資料/CAUNet-master/CapsNet/noisy_datafinal/PESQ.txt", "w")
        for k, (features, labels) in enumerate(eval_loader):
            features = features.cuda()  # [1, 1, num_frames,frame_size]
            labels = labels.cuda()  # [signal_len, ]

            output = net(features)  # [1, 1, sig_len_recover]
            output = output.squeeze()  # [sig_len_recover, ]

            # keep length same (output label)
            output = output[:labels.shape[-1]]

            eval_loss = torch.mean((output - labels) ** 2)
            total_eval_loss += eval_loss.data.item()

            est_sp = output.cpu().numpy()
            cln_raw = labels.cpu().numpy()

            save_path_noisy = 'D:/Desktop/kasuga/研究資料/CAUNet-master/CapsNet/noisy_datafinal'

            eval_metric = eval_composite(cln_raw, est_sp, sr)

            if eval_metric['pesq'] >= 3:
                quality = '/good'
            elif eval_metric['pesq'] < 2:
                quality = '/bad'
            else :
                quality = '/mid'

            est_sp = est_sp * 0.75 / np.max(est_sp)
            write(save_path_noisy + quality + '/noisy_' + str(count + 1) + '.wav', sr, est_sp)
            f.write(str(count + 1) + '\t'+ str(eval_metric['pesq']) + '\n')

            #st = get_stoi(cln_raw, est_sp, sr)
            #pe = get_pesq(cln_raw, est_sp, sr)
            #sn = snr(cln_raw, est_sp)
            total_pesq += eval_metric['pesq']
            total_ssnr += eval_metric['ssnr']
            total_stoi += eval_metric['stoi']
            total_cbak += eval_metric['cbak']
            total_csig += eval_metric['csig']
            total_covl += eval_metric['covl']

            count += 1
        avg_eval_loss = total_eval_loss / count
        f.close()

    return avg_eval_loss, total_stoi / count, total_pesq / count, total_ssnr / count, total_csig / count, total_cbak / count, total_covl / count

def test_run():
    test_data = EvalDataset(test_file_list_path_noisy, test_file_list_path_clean, frame_size=512, frame_shift=256)
    test_loader = DataLoader(test_data,
                               batch_size=1,
                               shuffle=False,
                               num_workers=2,
                               collate_fn=EvalCollate(),
                               pin_memory=True
                             )

    ckpt_path = 'D:/Desktop/kasuga/研究資料/CAUNet-master/CapsNet/model9/latest.model-92.model'

    model = Net()
    #model = nn.DataParallel(model, device_ids=[0, 1])
    checkpoint = Checkpoint()
    checkpoint.load(ckpt_path)
    model.load_state_dict(checkpoint.state_dict)
    model.cuda()
    print(checkpoint.start_epoch)
    print(checkpoint.best_val_loss)
    print(numParams(model))

    avg_eval, avg_stoi, avg_pesq, avg_ssnr, avg_csig, avg_cbak, avg_covl = evaluate(model, test_loader)

    #avg_stoi, avg_pesq, avg_ssnr, avg_cbak, avg_csig, avg_covl = eva_noisy(test_file_list_path)

    #print('Avg_loss: {:.4f}'.format(avg_eval))
    print('STOI: {:.4f}'.format(avg_stoi))
    print('SSNR: {:.4f}'.format(avg_ssnr))
    print('PESQ: {:.4f}'.format(avg_pesq))
    print('CSIG: {:.4f}'.format(avg_csig))
    print('CBAK: {:.4f}'.format(avg_cbak))
    print('COVL: {:.4f}'.format(avg_covl))

if __name__ == '__main__':
    test_run()