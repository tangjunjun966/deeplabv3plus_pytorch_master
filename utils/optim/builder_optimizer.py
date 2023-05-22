
import torch



def build_optimizer(optim_name,model,lr):

    optimizer=None
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4)
    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=5e-4)

    return optimizer



 # # Set up optimizer
 #    optimizer = torch.optim.SGD(params=[
 #        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
 #        {'params': model.classifier.parameters(), 'lr': opts.lr},
 #    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
 #    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
 #    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
 #    scheduler = None
 #    if opts.lr_policy == 'poly':
 #        scheduler = utils.PolyLR(optimizer, 30e3, power=0.9)
 #    elif opts.lr_policy == 'step':
 #        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
 #





