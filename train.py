import torch
from dataset import MapDataset, RefDataset
from utils import save_checkpoint, load_checkpoint, save_some_examples
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import testval
from encoder import StyleEncoder

def train_fn(disc, gen, encoder, src_loader, ref_loader, opt_disc, opt_gen, opt_en, l1, mse, epoch):
    Y_reals = 0
    Y_fakes = 0
    #loop = tqdm(src_loader, leave=True)
    iter_ref = iter(ref_loader)
    Y_real=0
    Y_fake=0
    num = 0
    gloss = 0
    loss_D1 = 0
    loss_G1 = 0
    loss_sty1 = 0
    loss_cyc1 = 0
    loss_id1 = 0
    for idx, (x, y) in enumerate(src_loader):

        x2, x3, y2 = next(iter_ref)
        #x2, y2 = next(iter_ref)
        src_img = x.to(config.DEVICE)
        src_label = y.to(config.DEVICE)
        ref_img = x2.to(config.DEVICE)
        ref_img2 = x3.to(config.DEVICE)
        ref_label = y2.to(config.DEVICE)

        # Train Discriminators H and Z
        
        D_real = disc(src_img, src_label)
        loss_real = mse(D_real, torch.ones_like(D_real))
        D_real2 = disc(ref_img, ref_label)
        loss_real2 = mse(D_real2, torch.ones_like(D_real2))
        D_real3 = disc(ref_img2, ref_label)
        loss_real3 = mse(D_real3, torch.ones_like(D_real3))
        #loss_reg = 
            
        with torch.no_grad():
            s_ref = encoder(ref_img, ref_label)
            ref_fake = gen(src_img, s_ref)
            
        
        D_fake = disc(ref_fake.detach(), ref_label.detach())
        loss_fake = mse(D_fake, torch.zeros_like(D_fake))
            
        D_loss = loss_real + loss_real2 + loss_real3 + loss_fake
            
            
        Y_reals += D_real.mean().item()
        Y_fakes += D_fake.mean().item()
            
        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train Generators H and Z
        
        # adversarial loss for both generators
        s_ref = encoder(ref_img, ref_label)
        ref_fake = gen(src_img, s_ref)
        D_fake = disc(ref_fake, ref_label)
        loss_G_fake = mse(D_fake, torch.ones_like(D_fake))
        #loss_G_fake = torch.mean(torch.abs(D_fake - torch.ones_like(D_fake)))
            
        #style reconstruction
        s_ref_fake = encoder(ref_fake, ref_label)
        loss_sty = l1(s_ref, s_ref_fake)
        #loss_sty = torch.mean(torch.abs(s_ref - s_ref_fake))
            
        #cycle loss
        s_src = encoder(src_img, src_label)
        src_cyc = gen(ref_fake, s_src)
        loss_cyc = l1(src_cyc, src_img)
        #loss_cyc = torch.mean(torch.abs(src_cyc - src_img))
            
        #identity loss
        #src_id = gen(src_img, s_src)
        #loss_id = l1(src_id, src_img)
        #loss_id = torch.mean(torch.abs(src_id - src_img))
        
        #d loss
        s_ref2 = encoder(ref_img2, ref_label)
        ref_fake2 = gen(src_img, s_ref2)
        ref_fake2 = ref_fake2.detach()
        loss_ds = l1(ref_fake2, ref_fake)
        
        # add all togethor
        G_loss = (
          loss_G_fake
          + loss_sty * config.LAMBDA_STYLE
          + loss_cyc * config.LAMBDA_CYCLE
          #+ loss_id * config.LAMBDA_IDENTITY
          + loss_ds
        )
        loss_D1 += D_loss.mean().item()
        loss_G1 += loss_G_fake.mean().item()
        loss_sty1 += loss_sty.mean().item()
        loss_cyc1 += loss_cyc.mean().item()
        #loss_id1 += loss_id.mean().item()
        
        gloss += G_loss.mean().item()
        
        opt_gen.zero_grad()
        opt_en.zero_grad()
        G_loss.backward()
        opt_gen.step()
        opt_en.step()
                 

        if idx % 200 == 0:
            save_image(src_img*0.5+0.5, f"./saved_images/{idx}src.png")
            save_image(ref_img*0.5+0.5, f"./saved_images/{idx}ref.png")
            save_image(ref_fake*0.5+0.5, f"./saved_images/{idx}fake.png")
            
            
        
        #loop.set_postfix(Y_real=Y_reals/(idx+1), Y_fake=Y_fakes/(idx+1), G_loss = gloss / (idx + 1))
        num = idx + 1
        
    print('Y_real=', Y_reals/num, 'Y_fake=', Y_fakes/num, 'G_loss=', gloss/ num)
    print('epoh=', epoch, 'loss_D=', loss_D1/num, 'loss_G=', loss_G1/num, 'loss_sty=', loss_sty1/num, 'loss_cyc=', loss_cyc1/num)
        


def main():
    disc = Discriminator().to(config.DEVICE)
    gen = Generator().to(config.DEVICE)
    encoder = StyleEncoder(style_dim=64, num_domains=3).to(config.DEVICE)
    
    opt_gen = optim.Adam(
        list(gen.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_disc = optim.Adam(
        list(disc.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_en = optim.Adam(
        list(encoder.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, disc, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_ENCODER, encoder, opt_en, config.LEARNING_RATE,
        )
    
    data_src = MapDataset(
        root=config.TRAIN_DIR_SRC, transform=config.transforms
    )
    data_ref = RefDataset(
        root=config.TRAIN_DIR_REF, transform=config.transforms
    )
    val_src = MapDataset(
       root=config.VAL_DIR_SRC, transform=config.transforms
    )
    val_ref = MapDataset(
       root=config.VAL_DIR_REF, transform=config.transforms
    )
    
    src_loader = DataLoader(
        data_src,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    ref_loader = DataLoader(
        data_ref,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    src_val = DataLoader(
        val_src,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    ref_val = DataLoader(
        val_ref,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )


    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, encoder, src_loader, ref_loader, opt_disc, opt_gen, opt_en,  L1, mse, epoch)
        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=f"./model/gen{epoch}.pth.tar")
            save_checkpoint(disc, opt_disc, filename=f"./model/critic{epoch}.pth.tar")
            save_checkpoint(encoder, opt_en, filename=f"./model/encoder{epoch}.pth.tar")
            
        save_some_examples(gen, encoder, src_val, ref_val, epoch, folder='./evaluation')
        
    testval.test_all_val(gen, encoder, src_val, ref_val, folder='./all_val')

if __name__ == "__main__":
    main()
    
    
