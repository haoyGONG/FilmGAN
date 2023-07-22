import torch
import config
from torchvision.utils import save_image
import os



def save_some_examples(gen, encoder, src_val, ref_val, epoch, folder):
    x, y = next(iter(src_val))
    x2, y2 = next(iter(ref_val))
    src_img, src_label = x.to(config.DEVICE), y.to(config.DEVICE)
    ref_img, ref_label = x2.to(config.DEVICE), y2.to(config.DEVICE)
    #在测试时把BN和dropout层给关闭
    gen.eval()
    encoder.eval()
    if not os.path.isdir(folder):
        os.makedirs(folder)
        
    with torch.no_grad():
        s_ref = encoder(ref_img, ref_label)
        ref_fake = gen(src_img, s_ref)
        
        #y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        #x = x * 0.5 + 0.5
        #print(torch.squeeze(y_fake, dim=0).cpu().numpy().shape)
        #y_fake = np.transpose(torch.squeeze(y_fake.cpu(), dim=0).numpy(), (1, 2, 0))
        #x = np.transpose(torch.squeeze(x.cpu(), dim=0).numpy(), (1, 2, 0))
        '''        
        fake_horse = fake_horse.cpu().squeeze(dim=0)
        x = x.cpu().squeeze(dim=0)
        
        
        fake_horse = fake_horse.detach().numpy()
        x = x.detach().numpy()
        
        fake_horse = np.transpose(fake_horse, (1, 2, 0))
        x = np.transpose(x, (1, 2, 0))
        
        fake_horse = Image.fromarray(np.uint8(fake_horse))
        x = Image.fromarray(np.uint8(x))
        
        fake_horse.save(f'./evaluation/gen_Z_{epoch}.jpg')
        x.save(f'./evaluation/input_{epoch}.jpg')
        '''
        save_image(src_img* 0.5 + 0.5, f'./evaluation/{epoch}_src.png')
        
        save_image(ref_img* 0.5 + 0.5, f'./evaluation/{epoch}_ref.png')
        
        save_image(ref_fake* 0.5 + 0.5, f'./evaluation/{epoch}_fake.png')
        

        #save_image(y_fake, folder + f"/y_gen_{epoch}.jpg")
        #save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.jpg")
        #save_image(x, folder + f"/input_{epoch}.jpg")
        #if epoch == 1:
            #save_image(y * 0.5 + 0.5, folder + f"./evaluation/label_{epoch}.jpg")
            
    #恢复BN和dropout
    gen.train()
    encoder.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



    
    
    
    
    
    
    