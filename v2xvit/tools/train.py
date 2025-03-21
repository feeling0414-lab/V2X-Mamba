import argparse
import os
import statistics
import time

import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import v2xvit.hypes_yaml.yaml_utils as yaml_utils
from v2xvit.tools import train_utils
from v2xvit.data_utils.datasets import build_dataset
#计算参数量
from fvcore.nn import FlopCountAnalysis, parameter_count_table
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")#合成数据生成
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true', help="whether train with half precision")#是否使用半精度
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              # batch_size=1,
                              num_workers=12,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            #batch_size=1,
                            num_workers=12,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(parameter_count_table(model))
    # # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # record training
    writer = SummaryWriter(saved_path)


    # half precision training自动混合精度训练，节省显存并加快推理速度
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()
    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    for epoch in range(init_epoch, max(epoches, init_epoch)):

        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        #进度条
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        for i, batch_data in enumerate(train_loader):

            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well

            if not opt.half:
                ouput_dict = model(batch_data['ego'])

                # start = time.time()
                # ouput_dict = model(batch_data['ego'])
                # end = time.time()
                # print("inference time = ",(end-start))
                # first argument is always your output dictionary,第一参数为模型输出
                # second argument is always your label dictionary.第二参数为标签

                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            else:

                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)
            # back-propagation
            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
        pbar2.close()
        #验证模型
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                total = 0
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    #ouput_dict = model(batch_data['ego'])

                    start = time.time()
                    ouput_dict = model(batch_data['ego'])
                    end = time.time()
                    #print("inference time :", (end - start))
                    # total = total+(end - start)
                    # if (i % 100)==0:
                    #     print(total/100)
                    #     total = 0

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                               valid_ave_loss))

            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)


        #保存模型
        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step()
    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
