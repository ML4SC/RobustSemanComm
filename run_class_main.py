import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import torch
import utils
import model   
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from engine import *
from pathlib import Path
from base_args import get_args
from datasets import build_dataset
from optim_factory import create_optimizer
from utils import get_model, sel_criterion, load_checkpoint
from utils import NativeScalerWithGradNormCount as NativeScaler


############################################################
def seed_initial(seed=0):
    seed += utils.get_rank()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args): 
    ### Configuration
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed_initial(seed=args.seed)
    print(args)
    ####################################### Get the model
    model = get_model(args)
    if args.resume:
        checkpoint_model = load_checkpoint(model, args)
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    patch_size = model.img_encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module  
    
    ##### Activate the proxy
    # proxy = get_model(args)
    # if args.resume:
    #     checkpoint_model = load_checkpoint(proxy, args)
    #     utils.load_state_dict(proxy, checkpoint_model, prefix=args.model_prefix)
    # proxy.to(device)
    # if args.distributed:
    #     proxy = torch.nn.parallel.DistributedDataParallel(proxy, device_ids=[args.gpu], find_unused_parameters=True)
    # wp_adver = WeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)
    # proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
    
    
    print("------------------------------------------------------")
    ############## Get the data and dataloader
    
    trainset = build_dataset(is_train=True, args=args)
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                            sampler=torch.utils.data.RandomSampler(trainset),
                                            num_workers=args.num_workers, pin_memory=True,
                                            batch_size=args.batch_size, shuffle=False)
    
    ############################################## Get the test dataloader
    
    valset = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(valset)
    
    if valset is not None:
        dataloader_val = torch.utils.data.DataLoader(
            valset, sampler=sampler_val, batch_size=int(1.0 * args.batch_size),
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    else:
        dataloader_val = None
    
    ############################# Get the optimizer and the other training settings
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(trainset) // total_batch_size

    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    
    ###################################################### Get the criterion
    criterion = sel_criterion(args).to(device)
    
    
    
    ################################## Auto load the model in the model record folder
    args.eval = False
    args.if_attack_test = False
    print(args.eval)
    print(args.if_attack_test)
    if args.eval:
        test_stats, latent, label = evaluate( net=model, dataloader=dataloader_val, 
                            device=device, criterion=criterion, train_type=args.train_type, if_attack=args.if_attack_test, plot=args.eval)
        print(test_stats)
        print(f"Accuracy of the network on the {len(valset)} test samples: {test_stats['acc']*100:.3f}")
        print(np.array(latent).shape)
        print(np.array(label).shape)

        flattened_data = latent.reshape(latent.shape[0],-1)

        # Perform PCA to reduce dimensions to 2
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(flattened_data)

        # Visualize the PCA results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=1, s=5, c=label, cmap='plasma')
        plt.colorbar(scatter, label='Labels')
        plt.title("PCA Visualization of Patch Embeddings")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        save_path = '/mnt/c/Users/npham3/RobustSemanComm/' + args.model + '.jpg'
        plt.savefig(save_path)
        exit(0)

    ################################## Start Training the T-DeepSC
    print(f"Start training for {args.epochs} epochs")
    max_accuracy = 0.0
    start_time = time.time()

    test_losses = []
    test_accuracies = []

    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Epochs"):
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)

        train_stats = train_epoch(
                model, criterion, trainloader, optimizer, device, epoch, loss_scaler, 
                args.train_type, args.if_attack_train, args.clip_grad, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, 
                update_freq=args.update_freq)
    
        if args.output_dir and args.save_ckpt:
            # print(epoch)
            # print(args.epochs)
            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=None)
        if dataloader_val is not None:
            test_stats = evaluate(net=model, dataloader=dataloader_val, 
                                device=device, criterion=criterion, train_type=args.train_type, if_attack=args.if_attack_test, plot=args.eval)
            print(f"Accuracy of the network on the {len(valset)} test images: {test_stats['acc']*100:.3f}")   
            test_accuracies.append(test_stats['acc']*100)   
            test_accuracies.append(test_stats['loss'])   
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.output_dir:
        test_stats_df = pd.DataFrame({
            # 'Epoch': range(1, len(test_losses) + 1),
            'Test Loss': test_losses,
            'Test Accuracy': test_accuracies
        })
        csv_file_path = f"{args.output_dir}/test_stats.csv"
        test_stats_df.to_csv(csv_file_path, index=False)
        print(f"Test statistics saved to {csv_file_path}")


if __name__ == '__main__':
    # import sys
    # print(sys.argv)
    opts = get_args()
    print(opts)
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
