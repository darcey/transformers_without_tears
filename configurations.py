import all_constants as ac


def base():
    config = {}

    config['embed_dim'] = 512
    config['ff_dim'] = 2048
    config['num_enc_layers'] = 6
    config['num_dec_layers'] = 6
    config['num_heads'] = 8

    # architecture
    config['use_bias'] = True
    config['fix_norm'] = True
    config['scnorm'] = True
    config['mask_logit'] = True
    config['pre_act'] = True

    config['clip_grad'] = 1.0
    config['lr_scheduler'] = ac.NO_WU
    config['warmup_steps'] = 8000
    config['lr'] = 3e-4
    config['lr_scale'] = 1.
    config['lr_decay'] = 0.8
    config['stop_lr'] = 5e-5
    config['eval_metric'] = ac.DEV_BLEU
    config['patience'] = 3
    config['alpha'] = 0.7
    config['label_smoothing'] = 0.1
    config['batch_size'] = 4096
    config['epoch_size'] = 1000
    config['max_epochs'] = 200
    config['dropout'] = 0.3
    config['att_dropout'] = 0.3
    config['ff_dropout'] = 0.3
    config['word_dropout'] = 0.1
    config['source_eos'] = True

    # Decoding
    config['decode_method'] = ac.BEAM_SEARCH
    config['decode_batch_size'] = 4096
    config['beam_size'] = 4
    config['max_parallel_beams'] = 0
    config['beam_alpha'] = 0.6
    config['use_rel_max_len'] = True
    config['rel_max_len'] = 50
    config['abs_max_len'] = 300
    config['allow_empty'] = False

    return config


def en_vi():
    config = base()
    config['epoch_size'] = 1500
    return config
