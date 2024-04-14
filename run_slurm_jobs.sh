chmod a+x *.sh


srun \
  --container-mounts=/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" \
  --container-workdir="`pwd`" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.08-py3.sqsh \
  --job-name=nlp-for-lr \
  --gpus=1 \
  --mem=32G \
  --cpus-per-task=4 \
  --task-prolog="`pwd`/install.sh" \
  sh train.sh
