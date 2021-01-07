import numpy as np
from model import GAN, discriminator_pixel, discriminator_image, discriminator_patch1, discriminator_patch2, generator, discriminator_dummy, pretrain_g
import utils
import os
from PIL import Image
import argparse
from keras import backend as K
import tensorflow as tf
from pathlib import Path

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
    FLAGS,_= parser.parse_known_args()
    # training settings 
    #os.environ['CUDA_VISIBLE_DEVICES']="2"
    n_rounds=10
    batch_size=FLAGS.batch_size
    patchSize = 128 ##Prebaked in what we made
    n_filters_d=32
    n_filters_g=32
    val_ratio=0.05
    init_lr=2e-4
    schedules={'lr_decay':{},  # learning rate and step have the same decay schedule (not necessarily the values)
            'step_decay':{}}
    alpha_recip=1./FLAGS.ratio_gan2seg if FLAGS.ratio_gan2seg>0 else 0
    rounds_for_evaluation=range(n_rounds)

    # set dataset
    dataset="CHASE"
    whole_img_size = (1024, 1024)
    img_size = (patchSize, patchSize)
    img_out_dir="{}/probability_maps".format(dataset)
    model_out_dir="{}/model_{}_{}".format(dataset,FLAGS.discriminator,FLAGS.ratio_gan2seg)
    auc_out_dir="{}/auc_{}_{}".format(dataset,FLAGS.discriminator,FLAGS.ratio_gan2seg)
    train_dir="/home/veljko/trainset"
    test_dir="/home/shared/retina/VGAN/aug/CHASE/test"
    p = Path(train_dir)
    if not os.path.isdir(img_out_dir):
        os.makedirs(img_out_dir)
    if not os.path.isdir(model_out_dir):
        os.makedirs(model_out_dir)
    if not os.path.isdir(auc_out_dir):
        os.makedirs(auc_out_dir)

    test_imgs, test_vessels, test_masks = utils.get_imgs(test_dir, False, whole_img_size, "CHASE", True)
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
    files = os.listdir(train_dir)
    n_train_imgs = len(files) * 64
    scheduler=utils.Scheduler(n_train_imgs//batch_size, n_train_imgs//batch_size, schedules, init_lr) if alpha_recip>0 else utils.Scheduler(0, n_train_imgs//batch_size, schedules, init_lr)
    print("training {} images :".format(n_train_imgs // 64))
    for n_round in range(n_rounds):
        print(f"Here I am at round {n_round}")
    # train D
        utils.make_trainable(d, True)
        it = iter(files)
        #for i in range(1):
        for i in range(scheduler.get_dsteps()):
            #print(f"At round {n_round} and step {i} of {scheduler.get_dsteps()} for D training.")
            data = np.load(str(p / next(it)))
            #print("I loaded images!")

            fake = g.predict(data[:batch_size,...,:3],batch_size=batch_size)
            #print("I made my generator prediction, God help me.")
            d_x_batch, d_y_batch = utils.input2discriminator(data[:batch_size,...,:3], np.expand_dims(data[:batch_size,...,4], 3), fake, d_out_shape)
            #print("I have discriminated!")
            loss, acc = d.train_on_batch(d_x_batch, d_y_batch)
            #print(f'Training done with {loss} and {acc}.')
            if i % 100 == 0:
                print(f'D{n_round}: {i} of {scheduler.get_dsteps()} with loss {loss} and acc {acc}')

        # train G (freeze discriminator)
        utils.make_trainable(d, False)
        it = iter(files)
        for i in range(scheduler.get_gsteps()):
        #for i in range(1):
            #print(f"At round {n_round} and step {i} of {scheduler.get_gsteps()} for G training.")
            data = np.load(str(p / next(it)))
            #print("I loaded images!")
            g_x_batch, g_y_batch=utils.input2gan(data[:batch_size,...,:3], np.expand_dims(data[:batch_size,...,4], 3), d_out_shape)
            #print("I made my generator input, God help me.")
            loss, acc = gan.train_on_batch(g_x_batch, g_y_batch)
            #print(f'Training done with {loss} and {acc}.')        
            if i % 100 == 0:
                print(f'G{n_round}: {i} of {scheduler.get_gsteps()} with loss {loss} and acc {acc}')
            
            # save the weights
    
        d.save_weights(os.path.join(model_out_dir,"d_{}_{}_{}.h5".format(n_round,FLAGS.discriminator,FLAGS.ratio_gan2seg)))
        g.save_weights(os.path.join(model_out_dir,"g_{}_{}_{}.h5".format(n_round,FLAGS.discriminator,FLAGS.ratio_gan2seg)))
        gan.save_weights(os.path.join(model_out_dir,"gan_{}_{}_{}.h5".format(n_round,FLAGS.discriminator,FLAGS.ratio_gan2seg)))

        
        # update step sizes, learning rates
        scheduler.update_steps(n_round)
        K.set_value(d.optimizer.lr, scheduler.get_lr())    
        K.set_value(gan.optimizer.lr, scheduler.get_lr())    
        
        if n_round in rounds_for_evaluation:
            test_imgs_bits = utils.chopup(test_imgs, 128)
            test_masks_bits = utils.chopup(test_masks, 128, True)
            test_vessels_bits = utils.chopup(test_vessels, 128, True)
            output = np.empty(test_vessels_bits.shape)
            for i in range(test_imgs_bits.shape[0]):
                if np.max(test_vessels_bits[i]) < 1:
                    print("Empty!")
                    continue
                generated=g.predict(test_imgs_bits[i],batch_size=batch_size)
                generated=np.squeeze(generated, axis=3)
                vessels_in_mask, generated_in_mask = utils.pixel_values_in_mask(test_vessels_bits[i], generated , test_masks_bits[i])
                auc_roc=utils.AUC_ROC(vessels_in_mask,generated_in_mask,os.path.join(auc_out_dir,"auc_roc_{}_{}.npy".format(n_round, i)))
                auc_pr=utils.AUC_PR(vessels_in_mask, generated_in_mask,os.path.join(auc_out_dir,"auc_pr_{}_{}.npy".format(n_round, i)))
                binarys_in_mask=utils.threshold_by_otsu(generated,test_masks_bits[i])
                dice_coeff=utils.dice_coefficient_in_train(vessels_in_mask, binarys_in_mask)
                acc, sensitivity, specificity=utils.misc_measures(vessels_in_mask, binarys_in_mask)
                utils.print_metrics(n_round+1, auc_pr=auc_pr, auc_roc=auc_roc, dice_coeff=dice_coeff, 
                                    acc=acc, senstivity=sensitivity, specificity=specificity, type='TESTING')
                
                # print test images
                segmented_vessel=utils.remain_in_mask(generated, test_masks_bits[i])
                output[i,...] = segmented_vessel[...]
            img = utils.integrate(output, True)
            #for index in range(segmented_vessel.shape[0]):
            for i in range(img.shape[0]):
                Image.fromarray((img[i]*255).astype(np.uint8)).save(os.path.join(img_out_dir,str(n_round)+"_segmented.png"))
