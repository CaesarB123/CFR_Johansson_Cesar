import tensorflow as tf
import numpy as np

from util import *

class cfr_net(object):
    """
    cfr_net implements the counterfactual regression neural network
    by F. Johansson, U. Shalit and D. Sontag: https://arxiv.org/abs/1606.03976

    This file contains the class cfr_net as well as helper functions.
    The network is implemented as a tensorflow graph. The class constructor
    creates an object containing relevant TF nodes as member variables.
    """

    def __init__(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims):

        # x: an input tensor
        # t: a tensor representing treatment
        # y_: a tensor representing the target output
        # p_t:
        # FLAGS: an object that contains various flags for the model
        # r_alpha: a tensor representing a regularization parameter alpha
        # r_lambda: a tensor representing a regularization parameter lambda
        # do_in: a scalar representing the dropout rate for input layer
        # do_out: a scalar representing the dropout rate for output layer
        # dims: a list representing the dimensions of the model

        self.variables = {}
        self.wd_loss = 0

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) #@TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """

        # x: an input tensor
        # t: a tensor representing treatment
        # y_: a tensor representing the target output
        # p_t:
        # FLAGS: an object that contains various flags for the model
        # r_alpha: a tensor representing a regularization parameter alpha
        # r_lambda: a tensor representing a regularization parameter lambda
        # do_in: a scalar representing the dropout rate for input layer
        # do_out: a scalar representing the dropout rate for output layer
        # dims: a list representing the dimensions of the model (D['dim'] , Dim_in = de dimensie van de input layer voor de representation layer
        #                                                                   Dim_out = de dimensie van de gewilde output van de representation layer

        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.r_alpha = r_alpha
        self.r_lambda = r_lambda
        self.do_in = do_in
        self.do_out = do_out

        dim_input = dims[0] # Het aantal variabelen van de inputdata
        dim_in = dims[1] # Het aantal dimensies van de input layer = 200
        dim_out = dims[2] # Het aantal dimensies gewenst na de representation layer = 100

        weights_in = []
        biases_in = []

        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel): # Hier gaan we na of het aantal van representation layer inputs is ofwel 0 : geen representation
            # layer of we hebben 1 representation layer, maar dan dient deze voor variable selection,
            # n_in = number of hidden layers needed in the representation layer = 3
            # varsel = (1) : variable selection wordt toegepast
            #          (0) : variable selection wordt niet toegepast
            dim_in = dim_input # Het aantal inputs van de inputdata, dus we overschrijven de data van eerder --> nu het aantal dimensies van de inputdata
        if FLAGS.n_out == 0:
            # n_out =  Het aantal hidden layers voor de regression = 3, als er geen regression layers gedefinieerd zijn
            if FLAGS.split_output == False: # Split_output = gaan we de output opsplitsen in treated en control? Zou dit niet TRUE moeten zijn?
                # Als we de output splitsen, dan moet de dimensie van de data toenemen, en als we dit niet doen, dan hoeft dit niet
                dim_out = dim_in+1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm: # Als we batch normalisation gaan toepassen, een manier om overfitting te vermijden, dan gaan we volgende twee zaken definiëren
            bn_biases = []
            bn_scales = []

        ''' Construct input/representation layers '''
        h_in = [x] # We create a hidden layer, met 25 variables, maar nog zonder instances, dus de first dimension is (None, Dim_input)
        for i in range(0, FLAGS.n_in): # We gaan dit n_in keer laten runnen : 1 keer voor elke input neuron (hier 3)
            if i==0: # Als de loop voor de eerste keer runt, dan gaan we dit doe
                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel: # Als we aan variable selection gaan doen
                    weights_in.append(tf.Variable(1.0/dim_input*tf.ones([dim_input]))) # deze command gaat een rescaling factor initialisen
                    # voor elke input feature, we maken een matrix met lengte dim_input, maar gaan al deze eentjes vermenigvuldigen met de gestandardiseerde
                    # value 1/dim_input, als we dit doen krijgen we een eerste layer met dim (25,None)
                else: # Als variabal selection niet TRUE is, dan gaan we een weight matrix creeërn met shape [dim_input (aantal dimensies van de input)
                    #, dim_in (het aantal dimensions of the input representation layer)] Het gaat deze matrix initiatiliseren met random normal distribution
                    # values, met standarddeviation FLAGS.weight_init, which is a commonly used scaling factor for weight initialization in deep learning.
                    # We hebben dus een matrix met dim_input rows en dim_in columns

                    weights_in.append(tf.Variable(tf.random.normal([dim_input, dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_input))))
                    # The variable is a weight matrix that is used to linearly transform the input layer to a hidden layer representation with dim_in units.
            else: # Als we niet meer in onze eerste neuron gaan, dan gaan we: een matrix maken van dim_in, dim_in (zie paint)
                weights_in.append(tf.Variable(tf.random.normal([dim_in,dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_in))))
                # We krijgen hierdoor een list of tensorflow variables : 3 neurons met elk dimensies (25,20,20,20) (zie Paint) (als we geen variable
                # selection doen: de output neurons van de current layer are the same as the neurons in the next layer

            ''' If using variable selection, first layer is just rescaling'''
            if FLAGS.varsel and i==0:
                biases_in.append([])
                h_in.append(tf.multiply(h_in[i],weights_in[i])) # We hebben de h[x] geinitiliseerd en gaan dit nu vermenigvuldigen met de weights
                # Dit zijn de rescaling weights dat we berekent hebben in de stap hierboven, we krijgen een h_in met als eerste de dimensie (None, 25) en
                # een tweede tensor met de eerste gegevens * de weights, dus gestandaardiseerd
                # Hier berekenen we niets, hier gaan we slechts de input wat gaan aanpassen
            else:
                biases_in.append(tf.Variable(tf.zeros([1,dim_in]))) # We zeggen dat alle biases een matrix zijn met 1 rij, allemaal 0'en = ( 0 0 0 0 ... )
                z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i] # Hier berekenen we effectief een nummer (we hebben hier een dimensie van None rows,
                # met 25 columns)

                if FLAGS.batch_norm: # Dit is volgens mij minder belangrijk : vorm van regularisation
                    batch_mean, batch_var = tf.nn.moments(z, [0])

                    if FLAGS.normalization == 'bn_fixed':
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                    else:
                        bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                        bn_scales.append(tf.Variable(tf.ones([dim_in])))
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

                h_in.append(self.nonlin(z)) # dit is een relu activation function, voegt een relu activated output toe aan onze x
                h_in[i+1] = tf.nn.dropout(h_in[i+1], do_in) # dit is een regularisation technique to prevent overfitting
        h_rep = h_in[len(h_in)-1] #Dit selecteert de representation van de list h, nadat deze de volledige stap is doorlopen

        if FLAGS.normalization == 'divide': # hier kunnen we de representation tensor van de code normaliseren
            h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keepdims=True))
        else:
            h_rep_norm = 1.0*h_rep # hier copieren we gewoon de representation tensor

        ''' Construct ouput layers '''
        y, weights_out, weights_pred = self._build_output_graph(h_rep_norm, t, dim_in, dim_out, do_out, FLAGS)
        # Nu geven we onze representation layer mee met de regression output, waardoor we de effectieve voorspellingen gaan doen
        #

        ''' Compute sample reweighting '''
        if FLAGS.reweight_sample: # wordt gebruikt om class imbalance aan te pakken
            w_t = t/(2*p_t) # the weight of the treated group, met p_t the probability of the true label, t represents a column vector with shape (m,1) with
            # m the number of samples
            w_c = (1-t)/(2*(1-p_t)) # the weight of the control group
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0 # als er geen normalisation wordt gedaan, dan gaan we de sample_weight op 1 houden

        self.sample_weight = sample_weight

        ''' Construct factual loss function '''
        if FLAGS.loss == 'l1':
            res = sample_weight*tf.abs(y_-y) # = mean absolute difference between the predicted value y_ en de actual values y, de res is calculated element-wise
            # for each sample in the tensors y_ en y
            risk = tf.reduce_mean(res) # hier nemen we het gemiddelde van alle res in de tensor om de risk te krijgen
            pred_error = tf.reduce_mean(res) # Dit is hetzelfde als de risk, we gaan dit steeds proberen te minimaliseren
        elif FLAGS.loss == 'log': # We gaan hier een negatieve log-likelhood berekenen
            y = 0.995/(1.0+tf.exp(-y)) + 0.0025 # Dit gaat een sigmoid correctie toepassen zodat de values van y_ zeker tussen 0 en 1 liggen
            res = y_*tf.math.log(y) + (1.0-y_)*tf.math.log(1.0-y) # de berekening van de log likelihood
            risk = -tf.reduce_mean(sample_weight*res)
            pred_error = -tf.reduce_mean(res)
        else: #mean squared error
            risk = tf.reduce_mean(sample_weight*tf.square(y_ - y))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

        ''' Regularization '''
        if FLAGS.p_lambda>0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.n_in):
                if not (FLAGS.varsel and i==0): # No penalty on W in variable selection
                    self.wd_loss += tf.nn.l2_loss(weights_in[i])

        ''' Imbalance error '''
        if FLAGS.use_p_correction: # niet gelijk aan 0
            p_ipm = self.p_t
        else:
            p_ipm = 0.5

        if FLAGS.imb_fun == 'mmd2_rbf': # Maximum Mean Discrepancy (MMD) measure with a radial basis function (RBF) kernel.
            imb_dist = mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma) # h_rep_norm : een normalised representation of the inputdata
            # t = treatment, it generates a difference metric between the original sample and the generated representations
            # is the MMD2 distance between the representations in h_rep_norm and the marginal distribution represented by p_ipm
            imb_error = r_alpha*imb_dist
        elif FLAGS.imb_fun == 'mmd2_lin': # It appears to be using the Maximum Mean Discrepancy (MMD) measure with a linear kernel.
            imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_error = r_alpha*mmd2_lin(h_rep_norm,t,p_ipm)
        elif FLAGS.imb_fun == 'mmd_rbf': # Maximum Mean Discrepancy (MMD) between two sets of features using a radial basis function (RBF) kernel.
            imb_dist = tf.abs(mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma))
            imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
        elif FLAGS.imb_fun == 'mmd_lin': # It appears to be using the Maximum Mean Discrepancy (MMD) measure with a linear kernel.
            imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
        elif FLAGS.imb_fun == 'wass':
            imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=False,backpropT=FLAGS.wass_bpt)
            imb_error = r_alpha * imb_dist
            self.imb_mat = imb_mat # FOR DEBUG
        elif FLAGS.imb_fun == 'wass2':
            imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=True,backpropT=FLAGS.wass_bpt)
            imb_error = r_alpha * imb_dist
            self.imb_mat = imb_mat # FOR DEBUG
        else:
            imb_dist = lindisc(h_rep_norm,p_ipm,t)
            imb_error = r_alpha * imb_dist

        ''' Total error ''' # the total error, which is the sum of the risk, imbalance error, and regularization term.
        tot_error = risk

        if FLAGS.p_alpha>0: # alleen als we hier rekening mee houden
            tot_error = tot_error + imb_error

        if FLAGS.p_lambda>0: # dit is voor regularisation.
            tot_error = tot_error + r_lambda*self.wd_loss


        if FLAGS.varsel: # we will select a subset
            self.w_proj = tf.compat.v1.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj) # We gaan de value van w_proj toewijzen aan het eerste element van weights_in
            # deze values gaan dus de eerste values van weights_in updaten. Weights_in[0] verwijst naar de eerste neuron, deze gaat dus puur de
            # variable selection doen

        self.output = y
        self.tot_loss = tot_error
        self.imb_loss = imb_error
        self.imb_dist = imb_dist
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm

    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input] #h_input is the input representation
        dims = [dim_in] + ([dim_out]*FLAGS.n_out) # the dimension we create a neural network with the 4 neurons (25,10,10,10) (zie paint)


        weights_out = []; biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(tf.random.normal([dims[i], dims[i+1]],stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                    'w_out_%d' % i, 1.0) # We gaan een neural network creëeren met random generated variables)
            weights_out.append(wo) #we create a neural network with (25,10,10,10)

            biases_out.append(tf.Variable(tf.zeros([1,dim_out]))) # we beginnen met een tensor van biases van 0  (0 0 0)
            z = tf.matmul(h_out[i],weights_out[i]) + biases_out[i] # dit is de output van de neural network
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlin(z)) # hier gaan we de relu activation function toepassen op de output om zo een output vector te krijgen
            h_out[i+1] = tf.nn.dropout(h_out[i+1], do_out) # hier gaan we de dropout functie doorvoeren, ter regularisation

        weights_pred = self._create_variable(tf.random.normal([dim_out,1],stddev=FLAGS.weight_init/np.sqrt(dim_out)), 'w_pred') # we creeeren een kolom vector met
        # de predicted weights, voorlopig gewoon een random normal generated kolom vector.
        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred') # we doen hetzelfde voor de predicted bias

        if FLAGS.varsel or FLAGS.n_out == 0: # Als we geen regression coefficient toepassen
            self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred) #we definieren een loss function en deze toepassen op de weights, we gaan deze loss proberen te minima
            #liseren

        ''' Construct linear classifier '''
        h_pred = h_out[1] # de predicted outcome
        y = tf.matmul(h_pred, weights_pred)+bias_pred # y is de effectieve output van het neurale model
        return y, weights_out, weights_pred

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''

        if FLAGS.split_output:

            i0 = tf.compat.v1.to_int32(tf.where(t < 1)[:,0]) # hier splitsen we de data van de control variables af
            i1 = tf.compat.v1.to_int32(tf.where(t > 0)[:,0]) # hier splitsen we de data op de treated variables

            rep0 = tf.compat.v1.gather(rep, i0) # Hier gaan we dit doorvoeren naar de representation
            rep1 = tf.compat.v1.gather(rep, i1) # Hier gaan we dit doorvoeren naar de representation (waarom hier geen rep_norm?)

            # Build the output voor beide representations afzonderlijk
            y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([i0, i1], [y0, y1]) # It combines the two outputs using tf.dynamic_stitch and returns the combined tensor y
            weights_out = weights_out0 + weights_out1 # as well as the sum of the weights_out and weights_pred variables.
            weights_pred = weights_pred0 + weights_pred1
        else:
            h_input = tf.concat([rep, t],1) # we gooien deze twee samen
            y, weights_out, weights_pred = self._build_output(h_input, dim_in+1, dim_out, do_out, FLAGS) # en dan gaan we de output maken met de
            # algemene gegevens

        return y, weights_out, weights_pred
