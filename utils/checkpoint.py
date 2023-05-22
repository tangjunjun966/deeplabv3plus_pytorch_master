import torch


def save_ckpt(path, model, optimizer, scheduler, epoch, best_miou, **kwargs):
    """
    save current model
    """
    save_info = {
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "best_score": best_miou,
    }
    for k, v in kwargs.items():
        save_info[k] = v
    torch.save(save_info, path)

    # torch.save({
    #     "cur_itrs": cur_itrs,
    #     "model_state": model.module.state_dict(),
    #     "optimizer_state": optimizer.state_dict(),
    #     "scheduler_state": scheduler.state_dict(),
    #     "best_score": best_score,
    # }, path)


def load_ckpt(ckpt_path, model):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_priormodel_ckpt(ckpt_path, model):
    '''
    训练后的权重重新载入模型，仅载入模型权重与ckpt对应一致的变量和shape的值
    '''
    print('\nloading ckpt weights ...')
    ckpt_weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    ckpt_weights = ckpt_weights['model_state']
    model_weights = model.state_dict()
    unload_weights = []
    for k in model_weights.keys():
        if k in ckpt_weights.keys():
            ckpt_shape = ckpt_weights[k].shape
            model_shape = model_weights[k].shape
            if ckpt_shape == model_shape:
                model_weights[k] = ckpt_weights[k]
            else:
                unload_weights.append(k)
    model.load_state_dict(model_weights, strict=True)
    print('\nunloading weights name:\n', unload_weights)
    return model




def load_resume_ckpt(ckpt_path, model):
    '''
    resume方式载入权重，必须保持权重与ckpt对应一致的变量和shape的值
    '''
    print('\nloading resume weights ...')
    ckpt_weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    ckpt_weights = ckpt_weights['model_state']
    model_weights = model.state_dict()
    unload_weights = []
    for k in model_weights.keys():
        if k in ckpt_weights.keys():
            ckpt_shape = ckpt_weights[k].shape
            model_shape = model_weights[k].shape
            if ckpt_shape == model_shape:
                model_weights[k] = ckpt_weights[k]
            else:
                unload_weights.append(k)
    model.load_state_dict(model_weights, strict=True)
    print('\nunloading weights name:\n', unload_weights)
    return model
