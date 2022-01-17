



#########################Pytorch Libraries###############################################
import torch
from torch.autograd import Variable


#############################################################################################

##########################Liberías complementarias###########################################

from collections import OrderedDict
from collections import namedtuple
import json

from itertools import product
from IPython.display import display, clear_output
import pytorch_model_summary as pms
import time
import pandas as pd
import numpy as np
from torchviz import make_dot
from tqdm.auto import tqdm

##############################################################################################

############################################################################################################
class RunBuilder():
    @staticmethod 
    def get_runs(params):
        
        Run=namedtuple('Run', params.keys())
        
        runs=[]
        for v in product(*params.values()):
            runs.append(Run(*v))
        
        return runs
"""
ENGLISH
This class allows the notebook to take a sequence of hyperparameters to prove in our training loop. The goal is to be performace-efficient, so we can 
take one tuple of parameters at once, without having problems with the memory.
 

"""
############################################################################################################
class RunManager():
    def __init__(self, batches_per_epoch=0): 
        """
        In order to have a better control of our trainings, 
        we need to define a class, which enables us to
        calls some methods step-by-step along the main code
        for the training loop, so we can change some parameters
        or correct some mistakes or bugs if needed, without
        any complication.

        We first define the __init__ method, in which we will
        add as argument the number of batches per epoch, which
        will be useful when we display the progress bar
        during the trainig. We also initiate some variables
        that will help us to control the trainig parameters.
        We specify some fo them next:

        * run_count= We would make a code, in order that we
        can run multiple tuples of hyperparemeters at the same
        time, without initializing again manually each run. We
        will call a "run" to every training process involving
        a specific tuple of hyperparameters. Every run will be
        registered as a numeric variable.

        *run_start_time: We will take the
        total time our machine takes to do the full run process.

        * epoch_start_time: inside each run, we will have, as
        natural, epochs for the training process. We will
        measure the time it takes the network to process
        each epoch individually, aside from the run time.

        * epoch_data= foe each epoch, along all the batches,
        we will store the hyperparameters that
        defines each run. We will save this data in this
        variable.

        * epoch_results: The metrics obtained in each epoch
        will be saved here.

        * run_results: By the end of each run, we will store
        some dictionaries with the information for every
        run we can train in a specific execution of the 
        notebook.

        * progress. This will be the progress bar we will create
        for each batch. That is why we need the batches_per_epoch
        parameter. We will explain deeply later how to achieve 
        this.

        * batches_per_epoch: In order to create the progress bar,
        we need to tell the bar over which parameter we want
        to increase the bar over time. This will be the batches.
        Once a batch is finished by the network, we will increase
        by one the bar. 

        """



        self.run_count = 0
        self.run_start_time = None        
        self.epoch_start_time = None
        self.epoch_count = 0
        self.epoch_data = None
        self.epoch_results = None
        self.run_results = []
        self.progress = None
        self.batches_per_epoch = batches_per_epoch
    
    def begin_run(self):
        
        """
        This method is simple: we just need to increase
        by 1 the number of the corresponding run, and we
        initialize the run time.
        """

        self.run_start_time=time.time()
        self.run_count += 1
        
    def begin_epoch(self):

        """
        In this method we only initialize the concrete
        parameters concerning the epochs. We take the first
        time for the epoch. Then we increase the epochs by 1.
        Next, we declare some OrderedDict variables for the
        data and results of the epochs.

        The final steps involve some class methods that we will
        explain later. 
        """

        self.epoch_start_time = time.time()        
        self.epoch_count += 1
        self.epoch_data = OrderedDict()
        self.epoch_results = OrderedDict()
        
        self.clear_displayed_results()
        self.display_progress()
        self.display_run_results()

    def track(self, key, value):

        """
        This method is crutial for our development. 
        We will use this method to track every metric
        we need in every epoch. For example, we will track
        the loss obtained of a network for every batch. We will
        store them in the epoch_data dictionary, and when we
        finish all the batches of the epoch, we will sum all the
        losses obtained, and then we will divide the sum by the
        number of epochs, so we get the mean loss.
        """

        if key not in self.epoch_data:            
            self.epoch_data[key] = [value]
        else:
            self.epoch_data[key].append(value)
        
    
    def add_result(self, key, value):
        """
        As we mentioned earlier, the final results of each
        epoch, like the loss, will be stored with this
        method in the epoch_results dictionary
        """

        self.epoch_results[key] = value

    
    def end_epoch(self):

        """
        Once we have covered all the batches of the dataloader,
        we need to track the final results in an Ordered dictionary.
        Once we get all the information, we need to store
        the data in the run_results final list. Also, we
        display them in a Pandas Data Frame, and we finish
        the progress of the bar.
        """


        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['epoch duration'] = time.time() - self.epoch_start_time
        results['run duration'] = time.time() - self.run_start_time
        
        for k, v in self.epoch_results.items(): results[k] = v
        
        self.run_results.append(results)
        self.progress.close()
        

    def end_run(self):

        """
        This is the final method we will call in our 
        training loop. We only need to call some other self
        methods that we will code later. We just erase the
        data we have displayed during the training progress,
        """

        self.clear_displayed_results()
        self.display_run_results()

    def display_progress(self):

        """
        We just initilize the progress bar that will tell
        us about how our net is doing with each batch.
        This is done by declaring two attributes: 
            - total: This is the number that the progress
            bar will look at when the net works along
            with the batches.
            - desc: This is just the text the bar will show
            every time the net gets through the different 
            epochs
        """

        self.progress = tqdm(
            total = self.batches_per_epoch,
            desc = f'Epoch {self.epoch_count} Progress'
        )

    def display_run_results(self):
        
        """
        We just display tha final results stored for each
        run 
        """
        
        if len(self.run_results) > 0:
            display(pd.DataFrame.from_dict(self.run_results, orient='columns'))
        
    def clear_displayed_results(self):

        """
        This method just clear whatever we have 
        printed in the console
        """

        clear_output(wait=True)
    
    def save(self, fileName): 

        """"
        Whatever we do during the training step, we would 
        like to save our information.
        """

        pd.DataFrame.from_dict(
            self.run_data
            ,orient='columns'
        ).to_csv(f'{fileName}.csv')
        
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)



    
#############################################################################################################
def plot_model(model, to_file,batch ,shapes ,show_shapes=True, show_layer_names=True, device='cuda'):
    
    """
    Since there is no inmediate way to show the plot of the
    model, as we can have with other frameworks, like 
    Keras, we want to emulate this same function with
    the following function.

    We first declare an empty variable, which will help
    us as input of the model. Then, we use the torchviz
    library to import the make_dot graph,so we can have a
    better visual representation of our graph.

    We declare twice the make_dot function, because we want
    to store the image in our files.
    """
    
    x=Variable(torch.rand(batch,*shapes)).to(device)
    y=model(x)
    make_dot(y.to('cpu'), params=dict(model.named_parameters()))
    dot=make_dot(y, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render(to_file[:-4])
    return dot

#############################################################################################################
def summary(Network, batch, dims, init_args=None, device='cuda'):    
    
    """
    We want also to recover the summary function from the
    Keras library. We print all the possible details of this
    particular function.

    We need to pass to the function the Network, batch size,
    dimensions of the input (channels, height and width),
    the initial arguments to declare the network, and 
    finally the device we are working with
    """
    
    batch_size=batch

    if init_args==None:
        print(pms.summary(Network(torch.zeros((batch_size, *dims))), 
                          show_input=True))
        print(pms.summary(Network().to(device), 
                          torch.zeros((batch_size,*dims)), show_input=False))
        print(pms.summary(Network().to(device), 
                          torch.zeros((batch_size,*dims)), show_input=False, show_hierarchical=True))
    else:
        print(pms.summary(Network(init_args).to(device), 
                          torch.zeros((batch_size, *dims)), show_input=True))
        print(pms.summary(Network(init_args).to(device), 
                          torch.zeros((batch_size,*dims)), show_input=False))
        print(pms.summary(Network(init_args).to(device), 
                          torch.zeros((batch_size,*dims)),
                          show_input=False,show_hierarchical=True))

"""
Esta función es análoga a la función con el mismo nombre del framework Keras
"""
#############################################################################################################
def gpu_arrow(device=0):
    t = torch.cuda.get_device_properties(device).total_memory
    a= torch.cuda.memory_summary()
    return (t-a)// 1073741824

"""
If we want to initialize some process in GPU, we could send all our data right from the beginning or
during the training step. We need to determine which method should be more useful for our purposes
"""
#############################################################################################################
