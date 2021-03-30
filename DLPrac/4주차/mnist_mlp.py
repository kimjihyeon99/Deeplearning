'''
Machine Learning Block Implementation Practice

Author : Sangkeun Jung (2019)
'''

# most of the case, you just change the component loading part
# all other parts are almost same
#

from mnist.rsc import load_rsc
from mnist.rsc import convert_to_tensor
from mnist.rsc import make_batch_data

from mnist.nn import mlp as network

import torch 
import torch.nn as nn
import numpy as np

fns = {
        'train' : 
        { 
            'image' : './mnist/data/train.image.npy',
            'label' : './mnist/data/train.label.npy'
        },
        'test' : 
        {
            'image' : './mnist/data/test.image.npy',
            'label' : './mnist/data/test.label.npy'
        },
        'model_fn' : './mnist/trained_model/mlp.model'
      }

def prepare_data(fn_dict, batch_size=100):
    """
        Three main components:
            1. load resource
            2. converting resource as tensor data
            3. batching
    """
    rsc           = load_rsc(fn_dict)
    converted_rsc = convert_to_tensor(rsc)
    batch_data    = make_batch_data(converted_rsc, batch_size)

    return batch_data

def train(model, batch_data, to_model_fn):
    model.train() # set information to pytorch that the current mode is 'training'

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  

    # training loop
    step = 0
    avg_losses = []
    for epoch in range(10):
        for idx, a_batch in enumerate(batch_data):
            batch_image, batch_label = a_batch
            
            batch_image = torch.tensor(torch.from_numpy(batch_image))
            batch_label = torch.tensor(torch.from_numpy(batch_label).type(torch.LongTensor))


            # init for updating the current batch
            optimizer.zero_grad()

            # forward pass
            predicts = model(batch_image)

            loss = criterion(predicts, batch_label)  # batch_label = reference
            
            # backward pass
            loss.backward() # accumulates the gradient (by addition) for each parameter. This is why you should call optimizer.zero_grad()

            # parameter update 
            optimizer.step()

            # monitoring at every 100 steps
            _loss = loss.item()  # loss.item() # gets the a scalar value held in the loss.
            avg_losses.append( _loss ) 
            if step % 100 == 0 :
                print('Epoch={} \t Step={} \t Loss={:.6f}'.format(
                            epoch, 
                            step,
                            np.mean(avg_losses)
                        )
                      )
                avg_losses = []
            step += 1

    # save model 
    import os
    _dir = os.path.dirname(to_model_fn)
    if not os.path.exists(_dir): os.makedirs(_dir)
    torch.save(model, to_model_fn)
    print("Model saved at {}".format(to_model_fn) )


def test(model, batch_data):
    model.eval() # set information to pytorch that the current mode is 'testing'

    all_predicts   = []
    all_references = []

    for idx, a_batch in enumerate(batch_data):
        batch_image, batch_label = a_batch
            
        batch_image = torch.tensor(torch.from_numpy(batch_image))
        batch_label = torch.tensor(torch.from_numpy(batch_label).type(torch.LongTensor))

        # forward pass
        predicts = model(batch_image) # predicts is logits
        logit = predicts.data.cpu().numpy()

        pred_image_idxs = np.argmax(logit, axis=1)

        for p in pred_image_idxs:all_predicts.append(p)
        for r in batch_label:    all_references.append(r)

    # calculate the accuracy
    num_corrects = 0
    for p,r in zip(all_predicts, all_references):
        if p == r : num_corrects += 1

    accuracy = float(num_corrects) / float( len(all_predicts) )
    print("Accuracy of the model on testing data : {:.6f}".format(accuracy))


if __name__ == '__main__':

    def train_mode():
        batch_data    = prepare_data(fns['train'], batch_size=100)
        model         = network()
        train(model, batch_data, fns['model_fn'])

    def test_mode():
        batch_data = prepare_data(fns['test'], batch_size=100)
        model = torch.load(fns['model_fn'])
        test(model, batch_data)

    train_mode()
    test_mode()



