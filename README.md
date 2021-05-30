# QAPMS
Question Answer Pipeline Model Serving

## Current models that are deployed in this project:

- **Question answer pipeline** (retriever + reader) (receive question choose relative articles then extract answer)
- **Standalone reader model** (reader) (receive question and context then extract answer from the context)

## Project structure

| File/Folder      | Description |
| ----------- | ----------- |
| QA      | the infos about the question answer pipeline (handler config utils ...)       |
| reader      | the infos about the reader model (handler config utils ...)       |
| docker   | docker files for dev and prod        |
| logs   | contains torchserve logs        |
| full_requirements.txt   | python requirements        |
| install_req   | script to install python environment dependencies        |
| prepare   | script to create .mar files in modelstore that are used for deployment        |
| prepare_prod   | script to download the model then prepare it  |
| run_dev   | script to run in development mode        |
| run_prod   | script to run in production mode        |
| stop   | script to stop all the working models    |

## How to run:
### normal mode:

1. Create a model store and model to download directory

`mkdir -p model_store`

2. Create python environment and install dependecies

`virtualenv env`

`source env/bin/activate`

`./install_req`

3. Prepare the .mar files

`./prepare` (or `./prepare_prod`)


4. Run the server

`./run_dev` (or `./run_prod`)

### Docker mode:

1. Build image

`docker build --tag QAPMS:0.1.0 -f docker/torchserve/DockerfileProd` (Or DockerfileDev)

2. run image

`docker run -p 80:80 QAPMS:0.1.0` (Expose the port you are using depending on dev and prod)

