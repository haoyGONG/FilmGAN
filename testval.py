from dataset import MapDataset
from utils import load_checkpoint
from torch.utils.data import DataLoader
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from encoder import StyleEncoder
from combine_result import combine


def test_all_val(gen, encoder, src_val, ref_val, folder):
    loop = tqdm(src_val, leave=True)
    iter_ref = iter(ref_val)
    
    gen.eval()
    encoder.eval()
    for idx, (x, y) in enumerate(loop):
        x2, y2 = next(iter_ref)
        src_img = x.to(config.DEVICE)
        src_label = y.to(config.DEVICE)
        ref_img = x2.to(config.DEVICE)
        ref_label = y2.to(config.DEVICE)
        
        s_ref = encoder(ref_img, ref_label)
        ref_fake = gen(src_img, s_ref)
        
        if src_label.item() == 0:
            src_class = 'gold200'
        elif src_label.item() == 1:
            src_class = 'hp5'
        elif src_label.item() == 2:
            src_class = 'portra160'
            
        if ref_label.item() == 0:
            ref_class = 'gold200'
        elif ref_label.item() == 1:
            ref_class = 'hp5'
        elif ref_label.item() == 2:
            ref_class = 'portra160'
        
        save_image(src_img* 0.5 + 0.5, f'./all_val/{idx}_src_{src_class}.png')
        
        save_image(ref_img* 0.5 + 0.5, f'./all_val/{idx}_ref_{ref_class}.png')
        
        save_image(ref_fake* 0.5 + 0.5, f'./all_val/{idx}_fake.png')
    
    gen.train()
    encoder.train()
    combine()



def main():
    gen = Generator(img_channels=3, num_residuals=6).to(config.DEVICE)
    encoder = StyleEncoder(style_dim=64, num_domains=3).to(config.DEVICE)
    
    opt_gen = optim.Adam(
        list(gen.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_en = optim.Adam(
        list(encoder.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_ENCODER, encoder, opt_en, config.LEARNING_RATE,
        )
        
    
    val_src = MapDataset(
       root=config.VAL_DIR_SRC, transform=config.transforms
    )
    val_ref = MapDataset(
       root=config.VAL_DIR_REF, transform=config.transforms2
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
        shuffle=False,
        pin_memory=True,
    )
    
    test_all_val(gen, encoder, src_val, ref_val, folder='./all_val')  
        


if __name__ == '__main__':
    main()
        
