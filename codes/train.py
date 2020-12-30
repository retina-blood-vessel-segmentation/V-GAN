import numpy as np
from model import GAN, discriminator_pixel, discriminator_image, discriminator_patch1, discriminator_patch2, generator, discriminator_dummy, pretrain_g
import utils
import os
from PIL import Image
import argparse
from keras import backend as K
import tensorflow as tf

with tf.device('/gpu:0'):
    # arrange arguments
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--ratio_gan2seg',
        type=int,
        help="ratio of gan loss to seg loss. Paper uses 10.",
        required=True
        )
    parser.add_argument(
        '--gpu_index',
        type=str,
        help="gpu index",
        required=True
        )
    parser.add_argument(
        '--discriminator',
        type=str,
        help="type of discriminator. Paper suggests image.",
        required=True
        )
    parser.add_argument(
        '--batch_size',
        type=int,
        help="batch size",
        required=True
        )
    parser.add_argument(
        '--dataset',
        type=str,
        help="dataset name",
        required=True
        )
    FLAGS,_= parser.parse_known_args()
    DRY = True
    # training settings 
    #os.environ['CUDA_VISIBLE_DEVICES']="2"
    n_rounds=10
    batch_size=FLAGS.batch_size
    n_filters_d=32
    n_filters_g=32
    val_ratio=0.05
    init_lr=2e-4
    schedules={'lr_decay':{},  # learning rate and step have the same decay schedule (not necessarily the values)
            'step_decay':{}}
    alpha_recip=1./FLAGS.ratio_gan2seg if FLAGS.ratio_gan2seg>0 else 0
    rounds_for_evaluation=range(n_rounds)

    # set dataset
    dataset=FLAGS.dataset
    img_size = (640, 640)
    if dataset == 'DRIVE':
        img_size = (640, 640)
    elif dataset == 'STARE':
        img_size = (720, 720)
    elif dataset == 'CHASE' or dataset == 'CHASEDB1':
        img_size = (1024, 1024)
    else:
        print("Unknown dataset. Panicking w/ arm-flailing.")
        exit(1)
    img_out_dir="{}/probability_maps".format(FLAGS.dataset)
    model_out_dir="{}/model_{}_{}".format(FLAGS.dataset,FLAGS.discriminator,FLAGS.ratio_gan2seg)
    auc_out_dir="{}/auc_{}_{}".format(FLAGS.dataset,FLAGS.discriminator,FLAGS.ratio_gan2seg)
    train_dir="/home/veljko/retina/repos/V-GAN/data/{}/train/".format(dataset)
    test_dir="/home/veljko/retina/repos/V-GAN/data/{}/test/".format(dataset)
    if not os.path.isdir(img_out_dir):
        os.makedirs(img_out_dir)
    if not os.path.isdir(model_out_dir):
        os.makedirs(model_out_dir)
    if not os.path.isdir(auc_out_dir):
        os.makedirs(auc_out_dir)

    trbatch = utils.LowMemoryTrainBatchFetcher(batch_size, train_dir, img_size, FLAGS.dataset, val_ratio, True, False)
    test_imgs, test_vessels, test_masks = utils.get_imgs(test_dir, False, img_size, "CHASE", True)
    g = generator(img_size, n_filters_g) 
    if FLAGS.discriminator=='pixel':
        d, d_out_shape = discriminator_pixel(img_size, n_filters_d,init_lr)
    elif FLAGS.discriminator=='patch1':
        d, d_out_shape = discriminator_patch1(img_size, n_filters_d,init_lr)
    elif FLAGS.discriminator=='patch2':
        d, d_out_shape = discriminator_patch2(img_size, n_filters_d,init_lr)
    elif FLAGS.discriminator=='image':
        d, d_out_shape = discriminator_image(img_size, n_filters_d,init_lr)
    else:
        d, d_out_shape = discriminator_dummy(img_size, n_filters_d,init_lr)

    utils.make_trainable(d, False)
    gan=GAN(g,d,img_size, n_filters_g, n_filters_d,alpha_recip, init_lr)
    generator=pretrain_g(g, img_size, n_filters_g, init_lr)
    g.summary()
    d.summary()
    gan.summary() 
    with open(os.path.join(model_out_dir,"g_{}_{}.json".format(FLAGS.discriminator,FLAGS.ratio_gan2seg)),'w') as f:
        f.write(g.to_json())

    with open(os.path.join(model_out_dir,"d_{}_{}.json".format(FLAGS.discriminator,FLAGS.ratio_gan2seg)),'w') as f:
        f.write(d.to_json())

    with open(os.path.join(model_out_dir,"gan_{}_{}.json".format(FLAGS.discriminator,FLAGS.ratio_gan2seg)),'w') as f:
        f.write(gan.to_json())
    # start training
    n_train_imgs = trbatch.getTrainN()
    scheduler=utils.Scheduler(n_train_imgs//batch_size, n_train_imgs//batch_size, schedules, init_lr) if alpha_recip>0 else utils.Scheduler(0, n_train_imgs//batch_size, schedules, init_lr)
    print("training {} images :".format(n_train_imgs))
    for n_round in range(n_rounds):
        print(f"Here I am at round {n_round}")
    # train D
        utils.make_trainable(d, True)
        for i in range(1):
        #for i in range(scheduler.get_dsteps()):
            #print(f"At round {n_round} and step {i} of {scheduler.get_dsteps()} for D training.")
            real_imgs, real_vessels = next(trbatch)
            #print("I loaded images!")

            fake = g.predict(real_imgs,batch_size=batch_size)
            #print("I made my generator prediction, God help me.")
            d_x_batch, d_y_batch = utils.input2discriminator(real_imgs, real_vessels, fake, d_out_shape)
            #print("I have discriminated!")
            loss, acc = d.train_on_batch(d_x_batch, d_y_batch)
            #print(f'Training done with {loss} and {acc}.')
            if i % 100 == 0:
                print(f'D{n_round}: {i} of {scheduler.get_dsteps()} with loss {loss} and acc {acc}')

        # train G (freeze discriminator)
        utils.make_trainable(d, False)
        #for i in range(scheduler.get_gsteps()):
        for i in range(1):
            #print(f"At round {n_round} and step {i} of {scheduler.get_gsteps()} for G training.")
            real_imgs, real_vessels = next(trbatch)
            #print("I loaded images!")
            g_x_batch, g_y_batch=utils.input2gan(real_imgs, real_vessels, d_out_shape)
            #print("I made my generator input, God help me.")
            loss, acc = gan.train_on_batch(g_x_batch, g_y_batch)
            #print(f'Training done with {loss} and {acc}.')        
            if i % 100 == 0:
                print(f'G{n_round}: {i} of {scheduler.get_gsteps()} with loss {loss} and acc {acc}')
            
            # save the weights
    
        d.save_weights(os.path.join(model_out_dir,"d_{}_{}_{}.h5".format(n_round,FLAGS.discriminator,FLAGS.ratio_gan2seg)))
        g.save_weights(os.path.join(model_out_dir,"g_{}_{}_{}.h5".format(n_round,FLAGS.discriminator,FLAGS.ratio_gan2seg)))
        gan.save_weights(os.path.join(model_out_dir,"gan_{}_{}_{}.h5".format(n_round,FLAGS.discriminator,FLAGS.ratio_gan2seg)))

            # evaluate on validation set
        if n_round in rounds_for_evaluation:
            if not trbatch.hasMoreValidation():
                trbatch.resetValidation()
            val_imgs, val_vessels = trbatch.getValidation()
            # D            
            d_x_test, d_y_test=utils.input2discriminator(val_imgs, val_vessels, g.predict(val_imgs,batch_size=batch_size), d_out_shape)
            loss, acc=d.evaluate(d_x_test,d_y_test, batch_size=batch_size, verbose=0)
            utils.print_metrics(n_round+1, loss=loss, acc=acc, type='D')
            # G
            gan_x_test, gan_y_test=utils.input2gan(val_imgs, val_vessels, d_out_shape)
            loss,acc=gan.evaluate(gan_x_test,gan_y_test, batch_size=batch_size, verbose=0)
            utils.print_metrics(n_round+1, acc=acc, loss=loss, type='GAN')
            
        
        # update step sizes, learning rates
        scheduler.update_steps(n_round)
        K.set_value(d.optimizer.lr, scheduler.get_lr())    
        K.set_value(gan.optimizer.lr, scheduler.get_lr())    
        
        if n_round in rounds_for_evaluation:
            generated=g.predict(test_imgs,batch_size=batch_size)
            generated=np.squeeze(generated, axis=3)
            vessels_in_mask, generated_in_mask = utils.pixel_values_in_mask(test_vessels, generated , test_masks)
            auc_roc=utils.AUC_ROC(vessels_in_mask,generated_in_mask,os.path.join(auc_out_dir,"auc_roc_{}.npy".format(n_round)))
            auc_pr=utils.AUC_PR(vessels_in_mask, generated_in_mask,os.path.join(auc_out_dir,"auc_pr_{}.npy".format(n_round)))
            binarys_in_mask=utils.threshold_by_otsu(generated,test_masks)
            dice_coeff=utils.dice_coefficient_in_train(vessels_in_mask, binarys_in_mask)
            acc, sensitivity, specificity=utils.misc_measures(vessels_in_mask, binarys_in_mask)
            utils.print_metrics(n_round+1, auc_pr=auc_pr, auc_roc=auc_roc, dice_coeff=dice_coeff, 
                                acc=acc, senstivity=sensitivity, specificity=specificity, type='TESTING')
            
            # print test images
            segmented_vessel=utils.remain_in_mask(generated, test_masks)
            for index in range(segmented_vessel.shape[0]):
                Image.fromarray((segmented_vessel[index,:,:]*255).astype(np.uint8)).save(os.path.join(img_out_dir,str(n_round)+"_{:02}_segmented.png".format(index+1)))