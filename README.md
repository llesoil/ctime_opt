# The Interplay of Compile-time and Run-time Options for Performance Prediction - Companion repository 

This is the companion repository of the SPLC'21 45th submission.

## Organization

Measurements and details about the performances can be consulted in the **data** folder.

Source code can be found in the **src** directory.

The **results** folder contains the results shown in the submission, as well as complementary results.

The submission **Interplay_compile_runtime.pdf** can be consulted in the root directory.

## Our research work in a nutshell

This paper investigates how compile-time options (top, in red on the following picture) can affect software performances (bottom, in black) and how compile-time options interact with run-time options (bottom, in green).

![picture](results/intro_fig_v4.png)

## Replication

To replicate our work, you have to:
1. Replicate the **measurement process**
2. **Run the code** in order to obtain our results

### Prerequisites

Install [docker](https://docs.docker.com/get-docker/). You can check that docker is working by checking its version (use the command line ```sudo docker --version```) or status (use ```sudo systemctl status docker```).

### 1. Measurement process

In this paper, we measured several performance properties of 4 different software systems (nodeJS, poppler, x264, and xz) for multiple run-time and compile-time configurations. 

For each of these system, we provide a docker container to measure its performances:
- for nodeJS, please follow this link : https://hub.docker.com/r/anonymicse2021/compile_nodejs
- for poppler, please follow this link : https://hub.docker.com/r/anonymicse2021/compile_poppler
- for x264, please follow this link : https://hub.docker.com/r/anonymicse2021/compile_x264
- for xz, please follow this link : https://hub.docker.com/r/anonymicse2021/compile_xz

For the rest of this part, we will consider x264's container, but the command lines we provide can be adapted to other containers. 

First, pull the container:
`sudo docker pull anonymicse2021/compile_x264`

Run it in interactive mode:
`sudo docker run -it anonymicse2021/compile_x264`

Now you are in the container, you can start the measurement process for x264. **WARNING! Since replicating all the measurements might take a while, we recommend you to stop the process after few minutes (2 or 3 is enough).**
`bash launchMeasures.sh`

To stop the process, just press ctrl + c. Now go to the output folder:
`cd output`
You will see a directory structure similar to the */data/x264/* directory (see also the explanations in *data/README.md*). To make sure that data are stored, just go in the first directory and display the first file:
`cd 1`
`cat original_videos_Gaming_360P_Gaming_360P-56fe.csv`

The first step is done!

### 2. Run the code

Now, you have the data. But how to replicate our results?

First, pull this container :


## Contact

If you have any question regarding this research paper or its replication, you can contact me at luc.lesoil@irisa.fr or via github, by creating an issue with your problem.


