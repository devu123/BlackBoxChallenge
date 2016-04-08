import interface as bbox
import theano
import numpy as np
import theano.tensor as T
import lasagne

def prepare_agent():
    net = lasagne.layers.InputLayer(shape=(None,1,1,n_features))
    net = lasagne.layers.DenseLayer(net,num_units=50,nonlinearity=lasagne.nonlinearity.tanh)
    net = lasange.layers.DenseLayer(net,num_units=n_actions,nonlinearity=lasagne.nonlinearity.softmax)
    return net

def get_all_scores(state,verbose=0):
    checkpoint_id = bbox.create_checkpoint()
    all_scores = np.array(1,n_actions)
    for a in range(n_actions):
        bbox.do_action(a)
        all_scores[a]=bbox.get_score()
        bbox.load_from_checkpoint(checkpoint_id)
    return all_scores

def get_action_by_state(state, verbose=0):
    if verbose:
        for i in range(n_features):
            print ("state[%d] = %f" %  (i, state[i]))

        print ("score = {}, time = {}".format(bbox.get_score(), bbox.get_time()))

    action_to_do = 0
    return action_to_do

n_features = n_actions = max_time = -1

 
def prepare_bbox():
    global n_features, n_actions, max_time
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
 
 
def run_bbox(verbose=False):
    has_next = 1
    
    prepare_bbox()
    agent=prepare_agent()

    while has_next:
        state = bbox.get_state()
        print np.shape(state)
        action = get_action_by_state(state)
        has_next = bbox.do_action(action)
 
    bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
    run_bbox(verbose=0)
 