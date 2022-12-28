from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def parse_args():
    import argparse

    parser = argparse.ArgumentParser('DDPM')
    # experimental settings
    parser.add_argument('--root_path', type=str, default='../../data/',
        help='Path of datasets.')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers for data loading (default: 4).')
    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, celeba, imagenet')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='MC steps.')


    # training settings
    parser.add_argument('--train_num_steps', type=int, default=700000,
        help='Number of steps for meta train.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--loss_type', type=str, default='l2',help='l1 or l2')
    parser.add_argument('--lr', type=float, default=2e-4)

    # network settings
    parser.add_argument('--hidden_dim', type=int, default=64)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    in_channel = 1 if args.dataset == 'mnist' else 3

    model = Unet(
        dim=args.hidden_dim,
        dim_mults=(1, 2, 4, 8),
        channels=in_channel
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,  # number of steps
        sampling_timesteps=250,
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type=args.loss_type,  # L1 or L2
        p2_loss_weight_gamma=1.
    ).cuda()

    trainer = Trainer(
        diffusion,
        None,
        root_folder=args.root_path,
        dataset_name=args.dataset,  # cifar10, mnist, celeba, imagenet
        num_workers=args.num_workers,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_num_steps=args.train_num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder='./results/'+args.dataset
    )

    trainer.train()
