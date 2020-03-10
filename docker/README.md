

## Setting up docker image
---
<!-- ### (Recommend)PULL DOCKER IMAGE
```
sudo docker pull eungbean/detectron2-vscode
``` -->

### BUILD DOCKER IMAGE
```
sudo docker build --tag eungbean/detectron2-vscode .
```

## DEPLOY DOCKER CONTAINER
---
* SET YOUR OWN PASSWORD WITH ```-e``` TAG
* BIND YOUR OWN DATAPATH WITH ```-v``` TAG
* SET GPUS WITH ```--gpus all``` or ```--gpus '"device=0,1"'``` TAG

```
sudo docker run --gpus all -it \
-p 5000:8080 -p 5006-5015:6006-6015 \
-v /path/to/dataset:/data/dataset \
-v /path/to/ADDrepo:/data/ADD \
--name ADD \
-e PASSWORD="0000" \
--ipc=host eungbean/detectron2-vscode /coder/code-server
```

<!-- ```
sudo docker run --gpus '"device=1"' -it \
-p 5000:8080 -p 5006-5015:6006-6015 \
-v /LIG/DATASET/ADD:/data/dataset \
-v /LIG/ADD/detectron2_ADD:/data/ADD \
--name ADD \
-e PASSWORD="0924" \
--ipc=host eungbean/detectron2-vscode /coder/code-server
``` -->

## Setting the CONTAINER Alias
---
echo "alias detectron2='docker exec -it detectron2 /bin/zsh'" >> ${ZDOTDIR:-$HOME}/.zshrc
source ~/.zshrc

<!-- 
echo "alias ADD='sudo docker exec -it ADD /bin/zsh'" >> ${ZDOTDIR:-$HOME}/.zshrc
source ~/.zshrc
 -->

 ## Build and install detectron2 inside the Docker
---
```
detectron2
# Now enter inside the docker container
# Current Work dir is /data/ADD

rm -rf build/ **/*.so
python3 -m pip install -e .
```
