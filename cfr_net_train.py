import cfr_net as cfr
import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback

tf.compat.v1.disable_eager_execution()

from util import *

# Er zijn nu 2 problemen waarmee ik zit:
#       (1) De iterations stoppen na 1-10 stappen, waarna we gewoon een 'nan' value krijgen en ik heb geen idee waarom dit gebeurt
#       (2) De distance tussen de verzamelingen (imb) is steeds 0, al weet ik niet waarom, de code lijkt te kloppen

# De code doorloopt volgende fases :
# STAP (1) - 606 : De run command start het proces
# STAP (2) - 368 : Voorbereidingen
        # (2.1) - 368 : Set up paths and start log - creëert de outputlogfiles
        # (2.2) - 396 : Load Data - laadt de data in, kijkt of het een .npz file is, kijkt of er een test_set aanwezig is ...
        # (2.3) - 419 : Start Session - De sessie wordt gestart. Nu zijn we in een dynamische omgeving
        # (2.4) - 422 : Initialise the input placeholders - creëert keys zodat we de data kunnen manipuleren
        # (2.5) - 439 : set up optimiser - Stelt het optimalisatie programma in, hier is er een foute mismatch met tensorflow V1 en V2 (PROBLEEM) -- HIER KAN IETS NIET KLOPPEN
        # (2.6) - 484 : Run for all repeated experiments : Volgende stappen gebeuren voor elk experiment
            # (2.6.1) - 494 : Load Data (if multiple repetitions, reuse first set) : We lezen alle data per experiment in, waardoor we steeds werken met 1 van de 100 experimenten in de dataset, we hebben 100 keer 700 instances, met elk 25 variabelen
            # (2.6.2) - 528 : Split into training and validation sets : We splitsen de dataset voor het geselecteerde experiment in Train en Valid
            # (2.6.3) - 546 : Run training loop : Run de training met al de selecteerde gegevens : CFR, sess, train_step, D_exp, I_valid, D_exp_test, logfile, i_exp - roepen STAP (3) op
            # (2.6.4) - 560 : Store predictions : Schrijf de gegenereerde output naar een excel document als deze flag aanstaat
            # (2.6.5) - 576 : Save results and predictions : Schrijf alle predictions en outcomes over naar de geselecteerde .npz file
# STAP (3) - 148 : Trainen per experiment
        # (3.1) - 151 : Train/validation split : Splits de data van dit bepaalde experiment op in train en validation set
        # (3.2) - 157 : Compute treatment probability : Genereer de prob op treated voor de dataset van dit experiment
        # (3.3) - 160 : Set up loss feed_dicts : Initialiseer de dictionarys van de training set, de validation set en de counterfactual set, zodat we later de berekeningen kunnen doen met deze data
        # (3.4) - 199 : Initialize TensorFlow variables : Bepaalt de beginwaarden van onze variabelen
        # (3.5) - 203 : Set up for storing predictions : Maak twee lists om onze predictions in op te slaan
        # (3.6) - 207 : Compute losses : Initialiseer de losses : bereken de losses zonder dat er optimalisatie wordt toegepast : voor zowel de factual training dictonary, the validation dictionary en als beschikbaar ook de counterfactual dictionary. Op het einde worden alle losses aan 1 list toegevoegd.
        # (3.7) - 252 : Train for multiple iterations : We gaan nu itereren per random geselecteerde batch
            # (3.7.1) - 256 : Fetch sample : select a random batch van de data waar we gaan op itereren
            # (3.7.2) - 266 : Do one step of gradient descent : Hierin gaan we minimaliseren op de losses, -- HIER KAN IETS NIET KLOPPEN
            # (3.7.3) - 281 : Compute loss every N iterations : We gaan de lossen berekenen per voor de totale dataset (niet meer de batch) en printen deze ook uit
            # (3.7.4) - 330 : Compute predictions every M iterations : We gaan de predictions berekenen voor de totale dataset van dit experiment (niet meer de batch) en slaan deze op

''' Define parameter flags '''  # Definiëren van de verschillende variabelen, zodat dit niet moet in de code
FLAGS = tf.compat.v1.app.flags.FLAGS  # De algemene verzameling van variabelen dat hieronder wordt gedefinieerd
tf.compat.v1.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.compat.v1.app.flags.DEFINE_integer('n_in', 3, """Number of representation layers. """)  # Definiëring het aantal van de hidden layers dat het model gaat
# gebruiken om de representation te maken.
tf.compat.v1.app.flags.DEFINE_integer('n_out', 3,"""Number of regression layers. """)  # Definiëring van het aantal regression layers :
# wordt gebruikt om regression te doen bv het schatten van een continious variable zoals de bloedspiegel. De prediction is produced by combining
# the outputs of the representation layers through a series of linear and nonlinear operations, represented by the weights and biases of the neurons.
# The prediction is then compared to the target value, and the difference between the prediction and target is used to train the network to produce
# better predictions. Deze regression layers zijn vaak verschillend van de output layers van bv classification methods.
tf.compat.v1.app.flags.DEFINE_float('p_alpha', 1e-4,"""Imbalance regularization parameter """)  # used to balance the distribution of the two datasets
# the discrepancy distance between te representations and the original data. Increasing the value of p_alpha makes the regularization term more important,
# which can improve the model's ability to handle imbalanced data. Dit moet veschillend zijn van 0 om MMD te laten meetellen
tf.compat.v1.app.flags.DEFINE_float('p_lambda', 1e-4,"""Weight decay regularization parameter. """)  # way of countering overfitting : The idea
# is that as the magnitude of the weights increase, the model becomes more complex and prone to overfitting, so adding a penalty term that
# decreases with the magnitude of the weights encourages the model to have smaller weights and thus reduce overfitting.
tf.compat.v1.app.flags.DEFINE_integer('rep_weight_decay', 0,"""Whether to penalize representation layers with weight decay""")  # Passen we de
# weight decay regularisation toe op de representation layer?
tf.compat.v1.app.flags.DEFINE_float('dropout_in', 0.3,"""Input layers dropout keep rate. """)  # hoeveel van de input behoud je?
tf.compat.v1.app.flags.DEFINE_float('dropout_out', 0.3,"""Output layers dropout keep rate. """)  # Hoeveel van de output behoud je?
tf.compat.v1.app.flags.DEFINE_string('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
tf.compat.v1.app.flags.DEFINE_float('lrate', 0.1, """Learning rate. """)
tf.compat.v1.app.flags.DEFINE_float('decay', 0.3,"""RMSProp decay. """)  # RMSProp is a variant of gradient descent optimization
# algorithm that uses moving averages of the gradient squared to scale the learning rate for each weight in the model.
# The decay rate determines the speed at which the historical information about the gradient squared is forgotten,
# with larger values meaning the moving average forgets older information more quickly.
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 10, """Batch size. """)
tf.compat.v1.app.flags.DEFINE_integer('dim_in', 20,"""Pre-representation layer dimensions. """)  # Het aantal neuronen dat zich bevinden per layer in de
# representation layer, we werken hier met 25 features, dus we transferren deze 25 naar 20.  We maken dus een representation van slechts 20 features
tf.compat.v1.app.flags.DEFINE_integer('dim_out', 10,"""Post-representation layer dimensions. """)  # Het aantal neuronen dat zich bevinden per layer in de
# regression layer. We gaan dan van 'dim_in' dimensions (hier 20) naar 10 neuronen, om hieruit 1 output te krijgen
tf.compat.v1.app.flags.DEFINE_integer('batch_norm', 0,"""Whether to use batch normalization. """)  # The goal of batch normalization is to stabilize the training
# process and prevent overfitting by normalizing the activations of neurons across each batch of data.
tf.compat.v1.app.flags.DEFINE_string('normalization', 'none',"""How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.compat.v1.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.compat.v1.app.flags.DEFINE_integer('experiments', 10,"""Number of experiments. """)  # Hoeveel experimenten zal je runnen? Ik denk dat hier een limiet
# op bestaat van 100, aangezien we maar met data van 100 experimenten werken
tf.compat.v1.app.flags.DEFINE_integer('iterations', 25,"""Number of iterations. """)  # Hoe vaak we geen repeaten in 1 experiment
# PROBLEEM : De error values zouden in mijn ogen moeten veranderen doorheen iteraties, maar dit gebeurt niet
tf.compat.v1.app.flags.DEFINE_float('weight_init', 0.1,"""Weight initialization scale. """)  # The weight initialization scale
# refers to the range of values used to initialize the weights of a neural network model before training. The purpose of weight
# initialization is to set the starting point for the optimization algorithm so that it can then learn the optimal weights for
# the problem at hand. The scale of the initialization values determines the magnitude of the weights at the start of training
# and can affect the speed and stability of the training process.
# A common practice is to initialize the weights with small random values, such as values drawn from a Gaussian distribution
# with a mean of 0 and a standard deviation of 0.1 or 0.01. This helps to prevent the optimization algorithm from getting stuck
# in suboptimal regions of the loss surface and encourages exploration of different parts of the parameter space.
tf.compat.v1.app.flags.DEFINE_float('lrate_decay', 0.97, """Decay of learning rate every 100 iterations """)
tf.compat.v1.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
tf.compat.v1.app.flags.DEFINE_float('wass_lambda', 10.0, """Wasserstein lambda. """)
tf.compat.v1.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)
tf.compat.v1.app.flags.DEFINE_integer('varsel', 0,"""Whether the first layer performs variable selection. """)  # The goal of variable selection is
# to identify the most important and relevant input variables that have a significant impact on the output prediction. = preprossessing
tf.compat.v1.app.flags.DEFINE_string('outdir', 'C:/Users/cesar/Documents/Cesar/Universiteit/Masterproef/outdir/',"""Output directory. """)
tf.compat.v1.app.flags.DEFINE_string('datadir', 'C:/Users/cesar/Documents/Cesar/Universiteit/Masterproef/datadir/',"""Data directory. """)
tf.compat.v1.app.flags.DEFINE_string('dataform', 'ihdp_npci_1-100.train.npz', """Training data filename form. """)
tf.compat.v1.app.flags.DEFINE_string('data_test', 'ihdp_npci_1-100.test.npz', """Test data filename form. """)
tf.compat.v1.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.compat.v1.app.flags.DEFINE_integer('seed', 0,"""Seed. """)  # Dit wordt gebruikt om dezelfde resultaten te bekomen zoals een voorganger
tf.compat.v1.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.compat.v1.app.flags.DEFINE_integer('use_p_correction', 0,"""Whether to use population size p(t) in mmd/disc/wass.""")
tf.compat.v1.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")  # The optimizer
# takes the gradients computed during backpropagation and uses them to update the weights in a way that reduces the loss and improves the model's performance.
# RMSProp: It is a variant of gradient descent optimization algorithm that uses moving averages of the gradient squared to scale the
# learning rate for each weight.
# Adagrad: It is an optimization algorithm that adapts the learning rate for each weight based on the historical gradient information.
# GradientDescent: It is the simplest optimization algorithm that updates the weights by subtracting the gradient of the loss with respect
# to the weights, multiplied by a learning rate.
# Adam: It is a popular optimization algorithm that combines ideas from gradient descent and RMSProp and is often used in deep learning applications.
tf.compat.v1.app.flags.DEFINE_string('imb_fun', 'mmd2_rbf',"""Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
# Welke penalty moeten we gebruiken om de imbalance van de normale verdeling en deze van de representation verdeling tegen te gaan
tf.compat.v1.app.flags.DEFINE_integer('output_csv', 1,"""Whether to save a CSV file with the results""")  # Moet je de data opslaan in een excel bestand?
tf.compat.v1.app.flags.DEFINE_integer('output_delay', 1,"""Number of iterations between log/loss outputs. """)  # om de hoeveel iteraties willen we een
# output ontvangen?
tf.compat.v1.app.flags.DEFINE_integer('pred_output_delay', 1,"""Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
# om de hoeveel iteraties willen we een prediction ontvangen?
tf.compat.v1.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.compat.v1.app.flags.DEFINE_integer('save_rep', 1,"""Save representations after training. """)  # Willen we deze representation opslaan?
tf.compat.v1.app.flags.DEFINE_float('val_part', 0.3,"""Validation part. """)  # Welk deel van de data gaan we gebruiken als validation? (0 < x < 1)
tf.compat.v1.app.flags.DEFINE_boolean('split_output', 0,"""Whether to split output layers between treated and control. """)
tf.compat.v1.app.flags.DEFINE_boolean('reweight_sample', 1,"""Whether to reweight sample for prediction loss with average treatment probability. """)
# Wordt gebruikt om class imbalance aan te pakken. wanneer de number of samples in each class is significantly different, In such cases,
# the classifier may be biased towards the class with more samples and have poor performance on the class with fewer samples. It increases the
# importance of the underrepresented class and down-weights the overrepresented class to ensure that both classes contribute equally to the loss function.

if FLAGS.sparse:  # Ik vermoed dat dit werkt als we een groot deel van de attributen als 0 laten
    import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 1  # Definiëring van deze term PROBLEEM: geen idee wat deze doet of verandert aan de outcome

__DEBUG__ = False  # Ik weet niet goed wat dit doet, maar vermoed dat dit niet zo belangrijk is
if FLAGS.debug:
    __DEBUG__ = True


# Counterfactual Regret Minimisation model (CFR)
# In a CFR model, the goal is to find a strategy for playing a game that minimizes the regret,
# which is the difference between the expected outcome of the best possible strategy and the expected outcome of the actual strategy.
# D : the train dataset D['x'] : de x-values (672,25), D['t'] : de treatment (672,1), D['y'] :  de outcome (672,1)
# I_valid : A set of instances (random set) which we will use for validation [410 430 356 130 ... 599 564 120]
# D_test : the test set : D_test['x'] : de x-values (672,25), D_test['t'] : de treatment (672,1), D_test['y'] :  de outcome (unknown) (672,1)
# logfile : the file to which the outputtext is written
# i_exp : the number of the experiment we are working : (0 < i_exp < experiments)
def train(CFR, sess, train_step, D, I_valid, D_test, logfile, i_exp):
    """ Trains a CFR model on supplied data """

    ''' Train/validation split '''  # Al deze zaken verschillen per experiment
    n = D['x'].shape[0]  # n = the number of instances = 672
    I = range(n)  # the range of instances (0,672)
    I_train = list(set(I) - set(I_valid))  # We nemen set (0,672) en trekken alle validation instances ervan af
    n_train = len(I_train)  # de grootte van de trainingset

    ''' Compute treatment probability'''  # Al deze zaken verschillen per experiment
    p_treated = np.mean(D['t'][I_train, :])  # We berekenen de kans om een treated instance te zijn in de trainset

    ''' Set up loss feed_dicts'''
    # Een feed dictionary wordt gebruikt om data te geven aan het model, waarbij de keys de placeholders zijn en de values de effectieve data
    dict_factual = {CFR.x: D['x'][I_train, :], CFR.t: D['t'][I_train, :], CFR.y_: D['yf'][I_train, :],
                    CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha,
                    CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}
    # into the model during model training
    # CFR.x = de placeholders voor de the covariates : de variabelen dat gebruikt worden om de dependent te verklaren (gewoon de extra attributen)
    #   D['x'][I_train,:] : alle 25 variables of the training instances :  worden gebruikt als input features om de outcome te voorspellen (D['y'])
    # CFR.t = de placeholder voor een treatment dataset
    #   D['t'][I_train,:] : de treatment variables van de training instances
    # CFR.y = de outcome
    #   D['yf'][I_train,:] : de factual outcomes van de training instances
    # CFR.do_in: = een dropout indicitor van hoeveel van de inputdata (in de representation layer moet je houden)
    # CFR.do_out = een dropout indiciator van hoeveel van de outputdata voor in de regression layer moeten we houden
    #   Dropout wordt gebruikt als regularisation om overfitting te vermijden, we gaan random 'activations' gaan uitschakelen om zo overfitting
    #   te vermijden, gaat de complexity doen dalen en zal zorgen dat we niet afhangen van 1 bepaalde zwaar wegende feature
    # CFR.r_alpha: used to balance the distribution of the two datasets : the discrepancy distance between te representations and the original data
    # CFR.r_lambda: Weight decay regularization parameter.
    # CFR.p_t: de megegeven kans op een treatment

    if FLAGS.val_part > 0:  # Als we een validation set hebben
        dict_valid = {CFR.x: D['x'][I_valid, :], CFR.t: D['t'][I_valid, :], CFR.y_: D['yf'][I_valid, :],
                      CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha,
                      CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}

    # Idem als hierboven, mits validation set

    if D['HAVE_TRUTH']:  # We controleren of de TRUE counterfactual aanwezig is in de dataset
        dict_cfactual = {CFR.x: D['x'][I_train, :], CFR.t: 1 - D['t'][I_train, :], CFR.y_: D['ycf'][I_train, :],
                         CFR.do_in: 1.0, CFR.do_out: 1.0}

        # CFR.x = de placeholders voor de the covariates : de variabelen dat gebruikt worden om de dependent te verklaren (gewoon de extra attributen)
        #   D['x'][I_train,:] : alle 25 variables of the training instances :  worden gebruikt als input features om de outcome te voorspellen (D['y'])
        # CFR.t = de placeholder voor een treatment dataset
        #   1 - D['t'][I_train,:] : de tegenovergestelde treatment variables van de training instances
        # CFR.y = de outcome
        #   D['cyf'][I_train,:] : de counterfactual outcomes van de training instances

    ''' Initialize TensorFlow variables '''
    sess.run(tf.compat.v1.global_variables_initializer())
    # command initializes the variables in your TensorFlow graph. This step is important because it sets the initial values for the variables that you defined in your graph.
    # initialiseert alle variables dat geintroduceerd werden in de Tensorflow graph dit commando moet runnen voordat we het model kunnen trainen

    ''' Set up for storing predictions '''
    preds_train = []  # Is de list van predictions voor de trainingset
    preds_test = []  # Is de ist van predictions voor de testset

    ''' Compute losses '''  # Gaat de losses berekenen gemaakt door het model
    losses = []
    obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], feed_dict=dict_factual)
    # Gaat variaben assignen aan de losses :
    # CFR.tot_loss: the total loss of the model, which is a weighted sum of the prediction loss and the regularization terms.
    #   obj_loss : is steeds anders per experiment
    # CFR.pred_loss: the prediction loss of the model, which measures the difference between the predicted output values and the actual output values.
    #   f_error : is steeds verschillend per experiment
    # CFR.imb_dist: the imbalance distance of the model, which measures the difference between the representation and actual data.
    # PROBLEEM : imb_err is steeds 0 voor elk experiment

    valid_obj = np.nan
    valid_imb = np.nan
    valid_f_error = np.nan  # we stellen de errors aan als niet bestaande
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                                                       feed_dict=dict_valid)
    # CFR.tot_loss: the total loss of the validation model, which is a weighted sum of the prediction loss and the regularization terms.
    #   valid_obj : is steeds anders per experiment
    # CFR.pred_loss: the prediction loss of the validation model, which measures the difference between the predicted output values and the actual output values.
    #   valid_f_error : is steeds verschillend per experiment
    # CFR.imb_dist: the imbalance distance of the validation model, which measures the difference between the representation and actual data.
    # print("valid_imb",valid_imb)
    # PROBLEEM : valid_imb is steeds 0 voor elk experiment

    cf_error = np.nan  # Als de counterfactuals niet beschikbaar zijn, dan gebruiken we hier gewoon np.NOT A NUMBER.
    if D['HAVE_TRUTH']:
        cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)
    # cf_error wordt steeds goed berekent, is steeds verschillend per experiment

    losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
    # We gaan steeds een lijstje van 7 elementen toekennen aan de losses Dit gebeurt 1 keer per experiment, waardoor we een uitgebreide lijst krijgen
    # obj_loss: the overall objective loss (combination of prediction error and imbalance distance)
    # for the factual predictions on the training set
    # f_error: the prediction error for the factual predictions on the training set
    # cf_error: the prediction error for the counterfactual predictions on the training set
    # (if the true counterfactual outputs are available)
    # imb_err: the imbalance distance for the factual predictions on the training set
    # valid_f_error: the prediction error for the factual predictions on the validation set
    # (if the validation set is available)
    # valid_imb: the imbalance distance for the factual predictions on the validation set
    # (if the validation set is available)
    # valid_obj: the overall objective loss (combination of prediction error and imbalance distance)
    # for the factual predictions on the validation set (if the validation set is available)

    objnan = False  # noem dit false

    reps = []  # lege reps set
    reps_test = []  # lege reps_test

    ''' Train for multiple iterations '''  # PROBLEEM, ik denk dat hier het schoentje wringt
    # Will run for FLAGS iterations keer
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        I = random.sample(range(0, n_train),
                          FLAGS.batch_size)  # I is een random sample size met dat FLAGS.batch_size groot is, afkomend van de trainingset
        x_batch = D['x'][I_train, :][I,
                  :]  # Van alle training samples in x, gaan we er random batch_size uithalen om verder mee te werken
        t_batch = D['t'][I_train, :][I]  # Van de random geselecteerde x's, gaan we ook de bijhorende t en yf ophalen
        y_batch = D['yf'][I_train, :][I]

        if __DEBUG__:  # niet echt van belang denk ik
            M = sess.run(cfr.pop_dist(CFR.x, CFR.t), feed_dict={CFR.x: x_batch, CFR.t: t_batch})
            log(logfile, 'Median: %.4g, Mean: %.4f, Max: %.4f' % (
            np.median(M.tolist()), np.mean(M.tolist()), np.amax(M.tolist())))

        ''' Do one step of gradient descent '''
        if not objnan:  # gaat 1 trainingstep toepassen met de data van de batch, als de obj false is
            sess.run(train_step, feed_dict={CFR.x: x_batch, CFR.t: t_batch, CFR.y_: y_batch,
                                            CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out,
                                            CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda,
                                            CFR.p_t: p_treated})
            # de feed data wordt deel van de feed_dict
            # in this code, the tensor "train_step" represents an optimization operation,
            # and the method "sess.run" performs that optimization by running it in the current session
            # with the specified feed data stored in the "feed_dict".

        ''' Project variable selection weights '''
        if FLAGS.varsel:  # PROBLEEM : Hier begrijp ik niet wat hij doet
            wip = simplex_project(sess.run(CFR.weights_in[0]), 1)
            sess.run(CFR.projection, feed_dict={CFR.w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i == FLAGS.iterations - 1:  # Will repeat every n iterations, gegeven door output_delay
            obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], feed_dict=dict_factual)
            # print("obj_loss",obj_loss)
            # PROBLEEM : hier vind hij slechts losses voor de eerste 1 - 5 iteraties, daarna stopt het met werken, heb geen idee waarom
            # We gaan sessrun gebruiken om de CFR.tot_loss, CFR.pred_loss, CFR.imb_dist tensors te evaluaten
            rep = sess.run(CFR.h_rep_norm, feed_dict={CFR.x: D['x'],
                                                      CFR.do_in: 1.0})  # de x values van de samples, en we laten geen data weg.
            # print(rep)
            # hier maken we een representation van de data, maar er moet volgens mij ergens een fout zitten, want de rep matrix is:
            # [[0. 0. 0. ... 0. 0. 0.]
            #  [0. 0. 0. ... 0. 0. 0.]
            #  [0. 0. 0. ... 0. 0. 0.]
            #  ...
            #  [0. 0. 0. ... 0. 0. 0.]
            #  [0. 0. 0. ... 0. 0. 0.]
            #  [0. 0. 0. ... 0. 0. 0.]]
            # Dit zou volgens mij niet mogen
            rep_norm = np.mean(np.sqrt(np.sum(np.square(rep),
                                              1)))  # We maken de L2 norm = square root of the sum of the squares, represents the length of a vector

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)
                # Hier merk ik hetzelfde probleem
                # print("cf_error",cf_error): steeds een waarde, maar deze waarde verandert niet meer na een aantal iteraties
                # berekent de loss van de counterfactual samples

            # berekent de lossfunction van de validationset
            valid_obj = np.nan;
            valid_imb = np.nan;
            valid_f_error = np.nan
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                                                               feed_dict=dict_valid)
                # print("valid_obj",valid_obj): steeds steeds een waarde, maar deze waarde verandert niet meer na een aantal iteraties

            # print per iteratie een mooie output van de gegenereerde losses
            losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
            loss_str = str(
                i) + '\tObj_loss: %.3f,\tf_error: %.3f,\tcf_error: %.3f,\timb_err: %.2g,\tvalid_f_error: %.3f,\tvalid_imb: %.2g,\tvalid_obj: %.2f' \
                       % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj)

            if FLAGS.loss == 'log':  # way of calculating the accuracy, using this log function
                y_pred = sess.run(CFR.output,
                                  feed_dict={CFR.x: x_batch, CFR.t: t_batch, CFR.do_in: 1.0, CFR.do_out: 1.0})
                y_pred = 1.0 * (y_pred > 0.5)
                acc = 100 * (1 - np.mean(np.abs(y_batch - y_pred)))  # y_batch are the ground labels
                loss_str += ',\tAccuracy: %.2f%%' % acc

            log(logfile, loss_str)  # schrijft de loss_str over naar de logfile dat wordt gegenereerd

            if np.isnan(obj_loss):  # als de variable (obj_loss) een NaN is, dan gaan we dit toevoegen aan de logfile
                log(logfile,
                    'Experiment %d: Objective is NaN. Skipping.' % i_exp)  # wordt meegegeven met de train operation
                objnan = True

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i == FLAGS.iterations - 1:
            y_pred_f = sess.run(CFR.output, feed_dict={CFR.x: D['x'], CFR.t: D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            # Dit werkt gewoon neit, maar ik denk dat er gewoon iets mis is met onze .run functie
            y_pred_cf = sess.run(CFR.output,
                                 feed_dict={CFR.x: D['x'], CFR.t: 1 - D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            preds_train.append(
                np.concatenate((y_pred_f, y_pred_cf), axis=1))  # Gaat de twee numpy arrays : y_pred_f en y_pred_cf
            # gaan combineren en gaat deze toevoegen aan de list preds_train: gaat de models prediction toevoegen aan een training set

            if D_test is not None:  # We gaan hetzelfde testen voor de testset
                y_pred_f_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], CFR.t: D_test['t'], CFR.do_in: 1.0,
                                                                CFR.do_out: 1.0})
                y_pred_cf_test = sess.run(CFR.output,
                                          feed_dict={CFR.x: D_test['x'], CFR.t: 1 - D_test['t'], CFR.do_in: 1.0,
                                                     CFR.do_out: 1.0})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test),
                                                 axis=1))  # gaat exact hetzelfde doen, maar voor de testset
                # en niet voor de trainingset zoals hierboven

            if FLAGS.save_rep and i_exp == 1:  # Dit gaat alleen gebruikt worden als we de eerste layer van representation layer will opslaan
                reps_i = sess.run([CFR.h_rep], feed_dict={CFR.x: D['x'], CFR.do_in: 1.0, CFR.do_out: 0.0})
                # print(reps_i)
                # [array([[0., 0., 0., ..., 0., 0., 0.],
                #        [0., 0., 0., ..., 0., 0., 0.],
                #        [0., 0., 0., ..., 0., 0., 0.],
                #        ...,
                #        [0., 0., 0., ..., 0., 0., 0.],
                #        [0., 0., 0., ..., 0., 0., 0.],
                #        [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)]
                # opnieuw krijgen we een volledige representation van 0'en
                reps.append(
                    reps_i)  # Hier creeëren we representations van de inputdata om deze later te gebruiken en voegen deze
                # toe aan een verzameling reps

                if D_test is not None:
                    reps_test_i = sess.run([CFR.h_rep], feed_dict={CFR.x: D_test['x'], CFR.do_in: 1.0, CFR.do_out: 0.0})
                    reps_test.append(reps_test_i)  # hier doen we hetzelfde voor de test set
    return losses, preds_train, preds_test, reps, reps_test


def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''  # maakt de logfile
    npzfile = outdir + 'result'
    npzfile_test = outdir + 'result.test'
    repfile = outdir + 'reps'
    repfile_test = outdir + 'reps.test'
    outform = outdir + 'y_pred'
    outform_test = outdir + 'y_pred.test'
    lossform = outdir + 'loss'
    logfile = outdir + 'log.txt'
    f = open(logfile, 'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    has_test = False
    if not FLAGS.data_test == '':  # if test set supplied
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test  # de file path for the test dataset is set to this name

    ''' Set random seeds '''  # dit genereert random variables, maar zorgt dat we deze herhalen waardoor we deze testen kunnen herhalen
    random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir + 'config.txt')

    log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha, FLAGS.p_lambda))

    ''' Load Data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile, 'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Start Session '''
    sess = tf.compat.v1.Session()

    ''' Initialize input placeholders '''
    x = tf.compat.v1.placeholder("float", shape=[None, D['dim']],
                                 name='x')  # Features #D[dim] is the number of features
    t = tf.compat.v1.placeholder("float", shape=[None, 1], name='t')  # Treatment
    y_ = tf.compat.v1.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    ''' Parameter placeholders '''
    r_alpha = tf.compat.v1.placeholder("float", name='r_alpha')
    r_lambda = tf.compat.v1.placeholder("float", name='r_lambda')
    do_in = tf.compat.v1.placeholder("float", name='dropout_in')
    do_out = tf.compat.v1.placeholder("float", name='dropout_out')
    p = tf.compat.v1.placeholder("float", name='p_treated')

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_in, FLAGS.dim_out]  # maakt de dimensies
    CFR = cfr.cfr_net(x, t, y_, p, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)  # geeft alle zaken mee naar cfr_net

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=True)  # mochten we dit naar TRUE overschakelen, dan zouden we
    # de value van deze varibalen updaten doorheen de training van het programma,
    # dit wordt niet vaak gebeurt aangezien we deze parameter dan kunnen gebruiken om te zien hoeveel stappen we gedaan hebben
    lr = tf.compat.v1.train.exponential_decay(FLAGS.lrate, global_step, NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay,
                                              staircase=True)
    # de learning rate gaat decaen na een bepaalde aantal iterations, wat ervoor zal zorgen dat we sneller convergeren en vermindert dus de
    # kans op overfitting The function computes an exponentially decaying learning rate based on the inputs,
    # which can be used to adjust the learning rate during training.

    opt = None  # Gaat de optimiser gaan selecteren die we willen gebruiken voor ons model
    if FLAGS.optimizer == 'Adagrad':
        opt = tf.compat.v1.train.AdagradOptimizer(lr)
    elif FLAGS.optimizer == 'GradientDescent':
        opt = tf.compat.v1.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'Adam':
        opt = tf.compat.v1.train.AdamOptimizer(lr)
    else:
        opt = tf.compat.v1.train.RMSPropOptimizer(lr, FLAGS.decay)

    ''' Unused gradient clipping '''
    # gvs = opt.compute_gradients(CFR.tot_loss)
    # capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
    # train_step = opt.apply_gradients(capped_gvs, global_step=global_step)

    train_step = opt.minimize(CFR.tot_loss, global_step=global_step)  # we try and minimse the loss function of CFR

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []
    if FLAGS.varsel:
        all_weights = None
        all_beta = None

    all_preds_test = []

    ''' Handle repetitions '''
    n_experiments = FLAGS.experiments
    if FLAGS.repetitions > 1:
        if FLAGS.experiments > 1:
            log(logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
            sys.exit(1)
        n_experiments = FLAGS.repetitions

    ''' Run for all repeated experiments '''
    for i_exp in range(1, n_experiments + 1):

        if FLAGS.repetitions > 1:
            log(logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
        else:
            log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))

        ''' Load Data (if multiple repetitions, reuse first set)'''

        if i_exp == 1 or FLAGS.experiments > 1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x'] = D['x'][:, :,
                             i_exp - 1]  # We all 672 entities, en alle 25 x values voor het eerste experiment
                # D[x][672,25,100] : dimensions of the datafile [subject, x values, experiment)
                # D[x][254,12,67] subject 254, feature 12, experiment 67 (tijd?)
                D_exp['t'] = D['t'][:,
                             i_exp - 1:i_exp]  # selecteert de treatment van alle personen voor i_exp is eig gewoon [d['t'][:,i_exp]], maar is beter voor debug
                D_exp['yf'] = D['yf'][:,
                              i_exp - 1:i_exp]  # selecteert de factual outcome van alle personen voor i_exp, is eig gewoon [d['yf'][:,i_exp]],
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:,
                                   i_exp - 1:i_exp]  # selecteert de counterfactual outcome van alle personen in i_exp, is eig gewoon [d['yf'][:,i_exp]],
                else:
                    D_exp['ycf'] = None  # selecteert de factual outcome van person i_exp,als zijnde None

                if has_test:  # We doen hier hetzelfde voor de test case van de data
                    D_exp_test = {}
                    D_exp_test['x'] = D_test['x'][:, :, i_exp - 1]
                    D_exp_test['t'] = D_test['t'][:, i_exp - 1:i_exp]
                    D_exp_test['yf'] = D_test['yf'][:, i_exp - 1:i_exp]
                    if D_test['HAVE_TRUTH']:
                        D_exp_test['ycf'] = D_test['ycf'][:, i_exp - 1:i_exp]
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform % i_exp
                D_exp = load_data(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)  # we splitsen de data op volgens de gegeven flag
        # I_train (met val_part 70%) : The rest of the data
        # I_valid (met val_part 30%) : [410 430 356 130 113 522 281 323 584 427 483 578  36 261 351 658 102 307
        #  561 614 368 106 363 376 450 566 487 671 499 192 394 277  95 536  74 299
        #  132 264  52 443 330   6 482 529 258 263 101 654 213 145 123 603  96 583
        #  341 629 548 616 412 342 352 203 609  64 136 178 620 553 565 535 232 663
        #  361 138 572  45 256 447 381 252 157 631 637 562 655 142  46 395 530 325
        #  579 666 362 595 625 108 489 266 278 194 471 316 488 398 274 291 496 357
        #  100 111 301 374 318 193 425 513 505 104 651 544 396 390 590  90 478  32
        #  375 345 110 144  87 268 283 403 365 140 512 437  69  49 475 626 149 324
        #  600 490 169 164 420 648 154 360 297 472 326  23  84 574 184 434 448  40
        #  652 433 440 502 459 607 509 415 649 611 656 377  78 399 340 257 191  19
        #  500 118 172  85  50 441   5 346 247 115 477 339  12 162  93 119   3 640
        #  599 564 120]

        ''' Run training loop '''  # we geven steeds alleen de dataset mee van de het experiment
        losses, preds_train, preds_test, reps, reps_test = train(CFR, sess, train_step, D_exp, I_valid, D_exp_test,
                                                                 logfile, i_exp)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)  # We gaan dit collecten voor alle experimenten
        all_preds_test.append(preds_test)  # we gaan dit collecten voor alle experimenten
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train, 1, 3), 0, 2)
        if has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test, 1, 3), 0, 2)
        out_losses = np.swapaxes(np.swapaxes(all_losses, 0, 2), 0, 1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)
        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (outform, i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (outform_test, i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (lossform, i_exp), losses, delimiter=',')

        ''' Compute weights if doing variable selection '''
        if FLAGS.varsel:
            if i_exp == 1:
                all_weights = sess.run(CFR.weights_in[0])
                all_beta = sess.run(CFR.weights_pred)
            else:
                all_weights = np.dstack((all_weights, sess.run(CFR.weights_in[0])))
                all_beta = np.dstack((all_beta, sess.run(CFR.weights_pred)))

        ''' Save results and predictions '''
        all_valid.append(I_valid)
        if FLAGS.varsel:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, w=all_weights, beta=all_beta,
                     val=np.array(all_valid))
        else:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test)

        ''' Save representations '''
        if FLAGS.save_rep and i_exp == 1:
            np.savez(repfile, rep=reps)

            if has_test:
                np.savez(repfile_test, rep=reps_test)


def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir + '/results_' + timestamp + '/'
    os.mkdir(outdir)

    try:
        run(outdir)
    except Exception as e:
        with open(outdir + 'error.txt', 'w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    tf.compat.v1.app.run()
