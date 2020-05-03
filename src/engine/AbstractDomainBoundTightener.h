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

#include "MStringf.h"
#include "NeuronIndex.h"
#include "PiecewiseLinearFunctionType.h"


#include "ap_global0.h"
#include "ap_global1.h"

#include "box.h"

#include "oct.h"
#include "pk.h"
#include "pkeq.h"

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

        // Step 2: propagate through the hidden layers
        for ( _currentLayer = 1; _currentLayer < _numberOfLayers; ++_currentLayer )
        {
            // Apply the weighted sum
            printf( "Starting affine transformation for layer %u\n", _currentLayer );
            performAffineTransformation();
            exit( 1 );

            // Apply the activation function
            applyActivationFunction();
        }

        deallocate();
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

    String weightedSumVariableToString( NeuronIndex index )
    {
        return Stringf( "ws_%u_%u", index._layer, index._neuron );
    }

    String activationResultVariableToString( NeuronIndex index )
    {
        return Stringf( "ar_%u_%u", index._layer, index._neuron );
    }

    void performAffineTransformation()
    {
        /*
          We create constraints that include:

            - The bounds for the previous layer
            - The variables of the current layer as a function of the
              activation results from the previous layer
        */

        unsigned previousLayerSize = (*_layerSizes)[_currentLayer - 1];
        unsigned currentLayerSize = (*_layerSizes)[_currentLayer];

        // Create the variable names and allocate the environment
        char **variables = new char *[previousLayerSize + currentLayerSize];
        for ( unsigned i = 0; i < previousLayerSize; ++i )
        {
            variables[i] = new char[12];
            String varName = activationResultVariableToString( NeuronIndex( _currentLayer - 1, i ) ).ascii();
            strncpy( variables[i], varName.ascii(), varName.length() );
        }
        for ( unsigned i = 0; i < currentLayerSize; ++i )
        {
            variables[i + previousLayerSize] = new char[12];
            String varName = weightedSumVariableToString( NeuronIndex( _currentLayer, i ) ).ascii();
            strncpy( variables[i + previousLayerSize], varName.ascii(), varName.length() );
        }

        ap_environment_t *apronEnvironment = ap_environment_alloc( NULL,
                                                                   0,
                                                                   (void **)&variables[0],
                                                                   previousLayerSize + currentLayerSize );

        ap_lincons1_array_t constraintArray = ap_lincons1_array_make( apronEnvironment, ( 2 * previousLayerSize ) + currentLayerSize );

        // Bounds for the previous layer
        for ( unsigned i = 0; i < previousLayerSize; ++i )
        {
            double lb = _lowerBoundsWeightedSums[_currentLayer - 1][i];
            double ub = _upperBoundsWeightedSums[_currentLayer - 1][i];

            ap_linexpr1_t expr = ap_linexpr1_make( apronEnvironment,
                                                    AP_LINEXPR_SPARSE,
                                                    1 );
            ap_lincons1_t cons = ap_lincons1_make( AP_CONS_SUPEQ,
                                                   &expr,
                                                   NULL );

            // ws - lb >= 0
            ap_lincons1_set_list( &cons,
                                  AP_COEFF_S_INT, 1, activationResultVariableToString( NeuronIndex( _currentLayer - 1, i ) ),
                                  AP_CST_S_DOUBLE, -lb,
                                  AP_END );

            ap_lincons1_array_set( &constraintArray, i * 2, &cons );

            // - ws + ub >= 0
            ap_lincons1_set_list( &cons,
                                  AP_COEFF_S_INT, -1, activationResultVariableToString( NeuronIndex( _currentLayer - 1, i ) ),
                                  AP_CST_S_DOUBLE, ub,
                                  AP_END );

            ap_lincons1_array_set( &constraintArray, i * 2 + 1, &cons );
        }

        // Weight equations
        for ( unsigned i = 0; i < currentLayerSize; ++i )
        {
            ap_linexpr1_t expr = ap_linexpr1_make( apronEnvironment,
                                                   AP_LINEXPR_SPARSE,
                                                   previousLayerSize + 1 );
            ap_lincons1_t cons = ap_lincons1_make( AP_CONS_EQ,
                                                   &expr,
                                                   NULL );

            // Add the target weighted sum variable and the bias
            ap_lincons1_set_list( &cons,
                                  AP_COEFF_S_INT, -1, weightedSumVariableToString( NeuronIndex( _currentLayer, i ) ).ascii(),
                                  AP_CST_S_DOUBLE, (*_bias)[NeuronIndex( _currentLayer, i )],
                                  AP_END );

            for ( unsigned j = 0; j < previousLayerSize; ++j )
            {
                double weight = _weights[_currentLayer - 1][j * currentLayerSize + i];
                ap_lincons1_set_list( &cons,
                                  AP_COEFF_S_DOUBLE, weight, weightedSumVariableToString( NeuronIndex( _currentLayer, i ) ).ascii(),
                                      AP_END );
            }

            // Register the constraint
            ap_lincons1_array_set( &constraintArray, previousLayerSize * 2 + i, &cons );
        }

        ap_abstract1_t av1 = ap_abstract1_of_lincons_array( _apronManager,
                                                            apronEnvironment,
                                                            &constraintArray );
        fprintf(stdout,"Affine transformation AV:\n");
        ap_abstract1_fprint( stdout, _apronManager, &av1 );

        for ( unsigned i = 0; i < previousLayerSize + currentLayerSize; ++i )
            delete[] variables[i];
        delete[] variables;

        ap_lincons1_array_clear( &constraintArray );
        ap_environment_free( apronEnvironment );
    }

    void applyActivationFunction()
    {
    }

    ap_manager_t *_apronManager;
    char *_apronVariables;

    void allocate()
    {
        _apronManager = box_manager_alloc();
    }

    void deallocate()
    {
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
