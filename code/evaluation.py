from Train import AverageMeter
import torch
import time

def validate(val_loader, model, criterion, epoch, args,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, user, target) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        user = user.cuda(non_blocking=True)
        target = torch.argmax(target.cuda(non_blocking=True),dim=1)
        output = model(input,user)
        target_var = torch.autograd.Variable(target)
        loss = criterion(output, target_var)
        losses.update(loss.item(), input.shape[0])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))

    logger.info(' * acc {acc.avg:.3f}'.format(acc=losses))
    return losses.avg