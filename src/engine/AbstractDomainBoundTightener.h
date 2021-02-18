/*********************                                                        */
/*! \file AbstractDomainBoundTightener.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#ifndef __AbstractDomainBoundTightener_h__
#define __AbstractDomainBoundTightener_h__

#include "Debug.h"
#include "MStringf.h"
#include "NeuronIndex.h"
#include "PiecewiseLinearFunctionType.h"


#include "ap_global0.h"
#include "ap_global1.h"

#include "box.h"
#include "t1p.h"
#include "oct.h"
#include "pk.h"
#include "pkeq.h"

#include "num.h"
#include "itv.h"
#include "t1p_internal.h"
#include "t1p_fun.h"

/*
  A superclass for performing abstract-interpretation-based bound
  tightening. This is a virtual class: a child class must provide the
  specific of the actual abstract domain that is being used (e.g.,
  box, polyhedron).
*/

class AbstractDomainBoundTightener
{
public:

    void initialize( unsigned numberOfLayers,
                     const Map<unsigned, unsigned> *layerSizes,
                     const Map<NeuronIndex, PiecewiseLinearFunctionType> *neuronToActivationFunction,
                     const double **weights,
                     const Map<NeuronIndex, double> *bias,
                     double **lowerBoundsWeightedSums,
                     double **upperBoundsWeightedSums,
                     double **lowerBoundsActivations,
                     double **upperBoundsActivations
    )
    {
        _numberOfLayers = numberOfLayers;
        _layerSizes = layerSizes;
        _neuronToActivationFunction = neuronToActivationFunction;
        _weights = weights;
        _bias = bias;

        _lowerBoundsWeightedSums = lowerBoundsWeightedSums;
        _upperBoundsWeightedSums = upperBoundsWeightedSums;
        _lowerBoundsActivations = lowerBoundsActivations;
        _upperBoundsActivations = upperBoundsActivations;
    }

    void run()
    {
        allocate();

        // Step 1: construct the input abstract value
        constructInputAbstractValue();

        // Step 2: propagate through the hidden layers
        for ( _currentLayer = 1; _currentLayer < _numberOfLayers ; ++_currentLayer )
        {
            performAffineTransformation();
            //performActivation();
      //      exit( 1 );
        }

        // TODO: handle output layer
        deallocate();
        exit(1);
    }

private:
    unsigned _numberOfLayers;
    const Map<unsigned, unsigned> *_layerSizes;
    const Map<NeuronIndex, PiecewiseLinearFunctionType> *_neuronToActivationFunction;
    const double **_weights;
    const Map<NeuronIndex, double> *_bias;

    double **_lowerBoundsWeightedSums;
    double **_upperBoundsWeightedSums;
    double **_lowerBoundsActivations;
    double **_upperBoundsActivations;

    unsigned _currentLayer;

    ap_abstract1_t _currentAV;

    String variableToString( NeuronIndex index )
    {
        return Stringf( "x_%u_%u", index._layer, index._neuron );
    }

    String weightedSumVariableToString( NeuronIndex index )
    {
        return Stringf( "ws_%u_%u", index._layer, index._neuron );
    }

    String activationResultVariableToString( NeuronIndex index )
    {
        return Stringf( "ar_%u_%u", index._layer, index._neuron );
    }

    void constructInputAbstractValue()
    {
        unsigned inputLayerSize = (*_layerSizes)[0];

        ap_lincons1_array_t constraintArray = ap_lincons1_array_make( _apronEnvironment,  2*inputLayerSize  );

        // Populate the constraints: lower and upper bounds
        for ( unsigned i = 0; i < inputLayerSize; ++i )
        {
            double lb = _lowerBoundsActivations[0][i];
            double ub = _upperBoundsActivations[0][i];
            printf("\n lower bound = %f ",lb);
            printf("\n upper bound = %f ",ub);

            // x - lb >= 0
            ap_linexpr1_t exprLb = ap_linexpr1_make( _apronEnvironment,
                                                     AP_LINEXPR_SPARSE,
                                                     1 );
            ap_lincons1_t consLb = ap_lincons1_make( AP_CONS_SUPEQ,
                                                     &exprLb,
                                                     NULL );

            printf("\n string name = %s",variableToString( NeuronIndex( 0, i ) ).ascii());

            ap_lincons1_set_list( &consLb,
                                  AP_COEFF_S_INT, 1, variableToString( NeuronIndex( 0, i ) ).ascii(),
                                  AP_CST_S_DOUBLE, -lb,
                                  AP_END );

            ap_lincons1_array_set( &constraintArray, i*2, &consLb );



            // - ws + ub >= 0
            ap_linexpr1_t exprUb = ap_linexpr1_make( _apronEnvironment,
                                                     AP_LINEXPR_SPARSE,
                                                     1 );
            ap_lincons1_t consUb = ap_lincons1_make( AP_CONS_SUPEQ,
                                                     &exprUb,
                                                     NULL );

            ap_lincons1_set_list( &consUb,
                                  AP_COEFF_S_INT, -1, variableToString( NeuronIndex( 0, i ) ).ascii(),
                                  AP_CST_S_DOUBLE, ub,
                                  AP_END );
            //printf("\n string name = %s ",variableToString( NeuronIndex( 0, i ) ).ascii());
            ap_lincons1_array_set( &constraintArray, ( i * 2 )+1 , &consUb );

        }

        // Extract the abstract value
        _currentAV = ap_abstract1_of_lincons_array( _apronManager,
                                                    _apronEnvironment,
                                                    &constraintArray );


        ap_lincons1_array_clear( &constraintArray );
    }

    void performAffineTransformation()
    {
        unsigned previousLayerSize = (*_layerSizes)[_currentLayer - 1];
        unsigned currentLayerSize = (*_layerSizes)[_currentLayer];

        // ap_lincons1_array_t constraintArray = ap_lincons1_array_make( _apronEnvironment, currentLayerSize );
        printf( "_currentAV after input layer(first layer) construction:\n");
        ap_abstract1_fprint( stdout, _apronManager, &_currentAV );

        // affine transformation
        for ( unsigned i = 0; i < currentLayerSize; ++i )
        {

            double lb = _lowerBoundsActivations[_currentLayer][i];
            double ub = _upperBoundsActivations[_currentLayer][i];
            printf("\n The Neuron name  = %s", variableToString( NeuronIndex( _currentLayer, i ) ).ascii());
            printf("\n lower bound = %f ",lb);
            printf("\n upper bound = %f ",ub);
            printf("\n Bias = %f ",(*_bias)[NeuronIndex( _currentLayer, i)]);
            ap_linexpr1_t expr1 = ap_linexpr1_make(_apronEnvironment,AP_LINEXPR_SPARSE,0);
            ap_linexpr1_t expr = ap_linexpr1_make(_apronEnvironment,AP_LINEXPR_SPARSE,0);
            for ( unsigned j = 0; j < previousLayerSize; ++j )
            {
                double weight = _weights[_currentLayer - 1][j * currentLayerSize + i];
                printf("\n weight = %f ",weight);
                ap_lincons1_set_list(reinterpret_cast<ap_lincons1_t *>(&expr1),
                                     AP_COEFF_S_DOUBLE, weight, variableToString( NeuronIndex( _currentLayer - 1, j ) ).ascii(),
                                     AP_END );
            }

            ap_linexpr1_set_list(&expr1,
                    //   AP_COEFF_S_INT, 1, "x_0_0",
                                 AP_CST_S_DOUBLE, (*_bias)[NeuronIndex( _currentLayer, i )],
                                 AP_END);
            fprintf(stdout,"\n Assignement (side-effect) in abstract value of neuron by expression:\n");
            ap_linexpr1_fprint(stdout,&expr1);
            _currentAV = ap_abstract1_assign_linexpr(_apronManager, true, &_currentAV,
                                                     (ap_var_t) variableToString(NeuronIndex(_currentLayer, i)).ascii(), &expr1, NULL);
            fprintf(stdout,"\n");

            fprintf(stdout," Current AV after equating weighted sum and bias :\n");
            //ap_abstract1_fprint(stdout,_apronManager,&_currentAV);
            fprintf(stdout,"\n");
            ap_linexpr1_clear(&expr1);
            ap_linexpr1_clear(&expr);
            // ******* Activation function *************

            // ***********Active case **************
            ap_lincons1_array_t arrayAct = ap_lincons1_array_make(_apronEnvironment, 1);
            ap_linexpr1_t exprAct = ap_linexpr1_make(_apronEnvironment, AP_LINEXPR_SPARSE, 2);
            ap_lincons1_t consAct = ap_lincons1_make(AP_CONS_SUPEQ, &exprAct, NULL);
            ap_lincons1_set_list(&consAct,
                                 AP_COEFF_S_DOUBLE, 1.0, variableToString(NeuronIndex(_currentLayer, i)).ascii(),
                                 AP_END);
            ap_lincons1_array_set(&arrayAct, 0, &consAct);
            ap_lincons1_array_fprint(stdout, &arrayAct);
            ap_abstract1_t activeAV = ap_abstract1_meet_lincons_array(_apronManager,false,&_currentAV,&arrayAct);
            fprintf(stdout,"Abstract value after activation (active case):\n");
            //ap_abstract1_fprint(stdout,_apronManager,&activeAV);
            ap_lincons1_array_clear( &arrayAct);
           // ap_linexpr1_clear(&exprAct);
           // ap_lincons1_clear(&consAct);

            /*
                Inactive case: neuron is zero
            */

            ap_lincons1_array_t arrayInAct = ap_lincons1_array_make(_apronEnvironment, 1);
            ap_linexpr1_t exprInAct = ap_linexpr1_make(_apronEnvironment, AP_LINEXPR_SPARSE, 2);
            ap_lincons1_t consInAct = ap_lincons1_make(AP_CONS_SUPEQ, &exprInAct, NULL);
            ap_lincons1_set_list(&consInAct,
                                 AP_COEFF_S_DOUBLE, -1.0, variableToString(NeuronIndex(_currentLayer, i)).ascii(),
                                 AP_END);
            ap_lincons1_array_set(&arrayInAct, 0, &consInAct);
            ap_lincons1_array_fprint(stdout, &arrayInAct);
            ap_abstract1_t inactiveAV = ap_abstract1_meet_lincons_array(_apronManager,false,&_currentAV,&arrayInAct);
            fprintf(stdout,"Abstract value after activation (inactive case):\n");
            ap_lincons1_array_clear( &arrayInAct);
           // ap_linexpr1_clear(&exprInAct);
           // ap_lincons1_clear(&consInAct);

            //ap_abstract1_fprint(stdout,_apronManager,&inactiveAV);

            /*
              Putting it all together: two meets and a join
            */
            bool notDestructive = false;
            ap_abstract1_t meetActive = ap_abstract1_meet( _apronManager,
                                                           notDestructive,
                                                           &activeAV,
                                                           &_currentAV );
            printf( "--- Applying activation function --- \n");
            printf( "meet active:\n" );
            //ap_abstract1_fprint( stdout, _apronManager, &meetActive );

            ap_abstract1_t meetInactive = ap_abstract1_meet( _apronManager,
                                                             notDestructive,
                                                             &_currentAV,
                                                             &inactiveAV );

            ap_linexpr1_t  inactiveExpr1= ap_linexpr1_make( _apronEnvironment,
                                                            AP_LINEXPR_SPARSE,
                                                            0 );
            // Neuron is negative
            ap_linexpr1_set_list(&inactiveExpr1,
                                 AP_COEFF_S_DOUBLE, 0.0, variableToString(NeuronIndex(_currentLayer, i)).ascii(),
                                 AP_END);
            meetInactive  = ap_abstract1_assign_linexpr(_apronManager, true, &meetInactive, (ap_var_t) variableToString(
                    NeuronIndex(_currentLayer, i)).ascii(), &inactiveExpr1, NULL);
            printf( "\nmeet inactive:\n" );
            //ap_abstract1_fprint( stdout, _apronManager, &meetInactive );
            _currentAV = ap_abstract1_join( _apronManager,
                                            notDestructive,
                                            &meetActive,
                                            &meetInactive );
            printf( "\nnew AV after join:\n" );
            printf(" The Neuron name  = %s", variableToString( NeuronIndex( _currentLayer, i ) ).ascii());
            printf("\n lower bound = %f ",lb);
            printf("\n upper bound = %f \n",ub);
            ap_abstract1_fprint( stdout, _apronManager, &_currentAV );


            /*
            // Weight equations
            ap_linexpr1_t  inactiveExpr1= ap_linexpr1_make( _apronEnvironment,
                                                            AP_LINEXPR_SPARSE,0 );
            // Neuron is negative
            ap_linexpr1_set_list(&inactiveExpr1,
                                 AP_COEFF_S_DOUBLE, 0.0, variableToString(NeuronIndex(_currentLayer, i)).ascii(),
                                 AP_END);

            ap_abstract1_t inactiveAV = ap_abstract1_assign_linexpr(_apronManager, true, &_currentAV, (ap_var_t) variableToString(NeuronIndex(_currentLayer, i)).ascii(), &inactiveExpr1, NULL);
            fprintf(stdout,"Abstract value: Inactive case\n");
            ap_abstract1_fprint(stdout,_apronManager,&inactiveAV);
            _currentAV = ap_abstract1_join(_apronManager,false,&inactiveAV,&activeAV);
            fprintf(stdout,"Abstract value 3 (join of 1 and 2):\n");
            ap_abstract1_fprint(stdout,_apronManager,&_currentAV);
        */
            printf("\n<--------------abstract test finished------------------>\n");


        }


    }

    void performActivation()
    {
        unsigned currentLayerSize = (*_layerSizes)[_currentLayer];

        for ( unsigned currentNeuron = 0; currentNeuron < currentLayerSize; ++currentNeuron )
        {
            /*
              Active case: neuron is positive and unchanged
            */

            ap_lincons1_array_t activeConstraintArray = ap_lincons1_array_make( _apronEnvironment, 1 );

            // Weight equations
            ap_linexpr1_t activeExpr = ap_linexpr1_make( _apronEnvironment,
                                                         AP_LINEXPR_SPARSE,
                                                         currentNeuron );
            ap_lincons1_t activeCons = ap_lincons1_make( AP_CONS_SUPEQ,
                                                         &activeExpr,
                                                         NULL );

            // Neuron is positive
            ap_lincons1_set_list( &activeCons,
                                  AP_COEFF_S_INT, 1, variableToString( NeuronIndex( _currentLayer, currentNeuron ) ).ascii(),
                                  AP_END );

            ap_lincons1_array_set( &activeConstraintArray, 0, &activeCons );

            ap_abstract1_t activeAV = ap_abstract1_of_lincons_array( _apronManager,
                                                                     _apronEnvironment,
                                                                     &activeConstraintArray );

            /*
              Inactive case: neuron is zero
            */

            ap_lincons1_array_t inactiveConstraintArray = ap_lincons1_array_make( _apronEnvironment, 1 );

            // Weight equations
            ap_linexpr1_t inactiveExpr = ap_linexpr1_make( _apronEnvironment,
                                                           AP_LINEXPR_SPARSE,
                                                           currentNeuron );
            ap_lincons1_t inactiveCons = ap_lincons1_make( AP_CONS_EQ,
                                                           &inactiveExpr,
                                                           NULL );

            // Neuron is positive
            ap_lincons1_set_list( &inactiveCons,
                                  AP_COEFF_S_INT, 1, variableToString( NeuronIndex( _currentLayer, currentNeuron ) ).ascii(),
                                  AP_END );

            ap_lincons1_array_set( &inactiveConstraintArray, 0, &inactiveCons );

            ap_abstract1_t inactiveAV = ap_abstract1_of_lincons_array( _apronManager,
                                                                       _apronEnvironment,
                                                                       &inactiveConstraintArray );

            /*
              Putting it all together: two meets and a join
            */
            bool notDestructive = false;

            ap_abstract1_t meetActive = ap_abstract1_meet( _apronManager,
                                                           notDestructive,
                                                           &_currentAV,
                                                           &activeAV );

            printf( "--- Applying activation function --- \n");

            printf( "meet active:\n" );
            //    ap_abstract1_fprint( stdout, _apronManager, &meetActive );

            ap_abstract1_t meetInactive = ap_abstract1_meet( _apronManager,
                                                             notDestructive,
                                                             &_currentAV,
                                                             &inactiveAV );

            printf( "\nmeet inactive:\n" );
            //    ap_abstract1_fprint( stdout, _apronManager, &meetInactive );


            _currentAV = ap_abstract1_join( _apronManager,
                                            notDestructive,
                                            &meetActive,
                                            &meetInactive );

            printf( "\nnew AV:\n" );
            //  ap_abstract1_fprint( stdout, _apronManager, &_currentAV );

            exit( 0 );
        }
    }

    ap_abstract1_t _wsAbstractValue;
    ap_manager_t *_apronManager;
    char *_apronVariables;

    char **_variableNames;
    unsigned _totalNumberOfVariables;

    ap_environment_t *_apronEnvironment;

    void allocate()
    {
        //_apronManager = box_manager_alloc();
       _apronManager = t1p_manager_alloc();

        // Count the total number of variables
        _totalNumberOfVariables = 0;
        for ( unsigned i = 0; i < _numberOfLayers; ++i )
            _totalNumberOfVariables += (*_layerSizes)[i];

        // Allocate the array
        _variableNames = new char *[_totalNumberOfVariables];

        // Populate the array
        unsigned counter = 0;
        for ( unsigned i = 0; i < _numberOfLayers; ++i )
        {
            for ( unsigned j = 0; j < (*_layerSizes)[i]; ++j )
            {
                NeuronIndex index( i, j );

                String varName = variableToString( index );
                _variableNames[counter] = new char[varName.length() + 1];
                strncpy( _variableNames[counter], varName.ascii(), varName.length() );
                _variableNames[counter][varName.length()] = '\0';
                ++counter;
            }
        }

        // std::cout<<"array value = ="<<_variableNames[0];

        _apronEnvironment = ap_environment_alloc( NULL,
                                                  0,
                                                  (void **)&_variableNames[0],
                                                  _totalNumberOfVariables );
    }

    void deallocate()
    {
        ap_environment_free( _apronEnvironment );

        for ( unsigned i = 0; i < _totalNumberOfVariables; ++i )
            delete[] _variableNames[i];
        delete[] _variableNames;

        ap_manager_free( _apronManager );
    }
};

#endif // __AbstractDomainBoundTightener_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
