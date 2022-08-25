from utils import my_parser, seed_util, wb_util, pair_selection_util
import os
from models.my_model import Encoder
import torch
from loaders import voxceleb_loader_for_pins
from configs.config import face_emb_dict, voice_emb_dict, emb_eva, model_save_folder
from utils.eval_shortcut import Cut


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data = data
    v_emb, f_emb = model(voice_data, face_data)
    loss = pair_selection_util.contrastive_loss(f_emb, v_emb, args.margin, tau_value)
    loss.backward()
    optimizer.step()
    info = {
        "train/tau_value": tau_value,
    }
    return loss.item(), info


def train():
    step = 0
    model.train()

    for epo in range(args.epoch):
        wb_util.log({"train/epoch": epo})
        for data in train_iter:
            loss, info = do_step(epo, step, data)
            step += 1
            if step % 50 == 0:
                obj = {
                    "train/step": step,
                    "train/loss": loss,
                }
                obj = {**obj, **info}
                print(obj)
                wb_util.log(obj)

            if step > 0 and step % args.eval_step == 0:
                if eval_cut.eval_short_cut():
                    return

            global tau_value
            if step > 0 and step % 500 == 0 and tau_value < 0.8:
                tau_value = tau_value + 0.1
                print("Update tau:", tau_value)


if __name__ == "__main__":
    parser = my_parser.MyParser(epoch=100, batch_size=64, model_save_folder=model_save_folder, early_stop=10)
    parser.custom({
        "batch_per_epoch": 500,
        "eval_step": 100,
        "margin": 0.6
    })
    parser.use_wb("PinsImpl", "run1")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    train_iter = voxceleb_loader_for_pins.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size, face_emb_dict, voice_emb_dict)

    tau_value = 0.3

    # model
    model = Encoder().cuda()
    model_params = model.parameters()

    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    eval_cut = Cut(emb_eva, model, args)
    train()
