

try:
    import openbayestool
except ImportError:
    openbayestool = None
else:
    from openbayestool import log_param, log_metric, clear_metric


if openbayestool:
    def log_train_acc(acc):
        log_metric('train acc max', max(acc))
        log_metric('train acc new', acc[-1])

    def log_test_acc(acc):
        log_metric('test acc max', max(acc))
        log_metric('test acc new', acc[-1])

    def log_args(param):
        log_param('epoch', param.epoch)
        log_param('n-way', param.n_way)
        log_param('k-spt', param.k_spt)
        log_param('k-qry', param.k_qry)
        log_param('img size', param.imgsz)
        log_param('img classes', param.imgc)
        log_param('task num', param.task_num)
        log_param('meta learning rate', param.meta_lr)
        log_param('update learning rate', param.update_lr)
        log_param('update step', param.update_step)
        log_param('update_step_test', param.update_step_test)
        log_param('data', param.data)


else:
    def log_train_acc(acc):
        pass

    def log_test_acc(acc):
        pass

    def log_param(param):
        pass
