

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

<!-- ## Test docker image
---
```
sudo docker run --rm --gpus all eungbean/detectron2-vscode nvidia-smi
``` -->

## DEPLOY DOCKER CONTAINER
---
* SET YOUR OWN PASSWORD WITH ```-e``` TAG
* BIND YOUR OWN DATAPATH WITH ```-v``` TAG

```
sudo docker run --gpus all -it \
-p 7100:8080 -p 7106-7115:6006-6015 \
-v /path/to/dataset:/data/dataset \
-v /path/to/addrepo:/data/ADD_repo \
--name detectron2 \
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