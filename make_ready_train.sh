#!/bin/bash
#Script needed to make this folder, once checked out, ready for use on any of the cluster computers.
# It will create the symlinks needed for all the data, the models and all that. 
#ln -s /home/shared/retina/models/iternet/trained models
ln -s /home/shared/retina/IterNet/data data
ln -s /home/shared/retina/output/vgan results