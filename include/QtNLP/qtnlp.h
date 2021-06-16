/****************************************************************************
 *                                                                          *
 * Copyright (C) 2015 Neutrino International Inc.                           *
 *                                                                          *
 * Author : Brian Lin <lin.foxman@gmail.com>, Skype: wolfram_lin            *
 *                                                                          *
 ****************************************************************************/

#ifndef QT_NLP_H
#define QT_NLP_H

#include <QtCore>
#include <QtNetwork>
#include <QtSql>
#include <QtScript>
#include <Essentials>

QT_BEGIN_NAMESPACE

#ifndef QT_STATIC
#    if defined(QT_BUILD_QTNLP_LIB)
#      define Q_NLP_EXPORT Q_DECL_EXPORT
#    else
#      define Q_NLP_EXPORT Q_DECL_IMPORT
#    endif
#else
#    define Q_NLP_EXPORT
#endif

namespace N
{

/*****************************************************************************
 *                                                                           *
 *                        Computational Linguistics                          *
 *                                                                           *
 *                 Natural Language Processing and Corpus                    *
 *                                                                           *
 *****************************************************************************/

namespace NLP
{

/*****************************************************************************
 *                                                                           *
 *                        Sequential Taggers for NLP                         *
 *                                                                           *
 *****************************************************************************/

class Q_NLP_EXPORT LrNode      ;
class Q_NLP_EXPORT LrProblem   ;
class Q_NLP_EXPORT LrParameter ;
class Q_NLP_EXPORT LrFunction  ;
class Q_NLP_EXPORT LrL2R       ;
class Q_NLP_EXPORT LrL2Svc     ;
class Q_NLP_EXPORT LrL2Svr     ;
class Q_NLP_EXPORT LrSolver    ;
class Q_NLP_EXPORT LrTron      ;
class Q_NLP_EXPORT Linear      ;

class Q_NLP_EXPORT LrNode
{
  public:

    int    index ;
    double value ;

    explicit LrNode     (void) ;
             LrNode     (const LrNode & node) ;
    virtual ~LrNode     (void) ;

    LrNode & operator = (const LrNode & node) ;

  protected:

  private:

};

class Q_NLP_EXPORT LrProblem
{
  public:

    int       l    ;
    int       n    ;
    double    bias ; // < 0 if no bias term
    double  * y    ;
    LrNode ** x    ;

    explicit LrProblem (void) ;
    virtual ~LrProblem (void) ;

    void     Camp      (void) ;

  protected:

  private:

};

class Q_NLP_EXPORT LrParameter
{
  public:

    typedef enum            {
      L2R_LR                   ,
      L2R_L2LOSS_SVC_DUAL      ,
      L2R_L2LOSS_SVC           ,
      L2R_L1LOSS_SVC_DUAL      ,
      MCSVM_CS                 ,
      L1R_L2LOSS_SVC           ,
      L1R_LR                   ,
      L2R_LR_DUAL              ,
      L2R_L2LOSS_SVR      = 11 ,
      L2R_L2LOSS_SVR_DUAL      ,
      L2R_L1LOSS_SVR_DUAL      }
      SolverTypes              ; /* solver types */

    int      Solver      ;
    double   eps         ;
    double   C           ;
    int      NrWeight    ;
    int    * WeightLabel ;
    double * Weight      ;
    double   p           ;

    explicit      LrParameter      (void) ;
                  LrParameter      (const LrParameter & param) ;
    virtual      ~LrParameter      (void) ;

    LrParameter & operator =       (const LrParameter & param) ;
    QString       Check            (void) ;

    bool          isSVR            (void) ;
    bool          ProbabilityModel (void) ;

  protected:

  private:

};

class Q_NLP_EXPORT LrFunction
{
  public:

    explicit       LrFunction    (void) ;
    virtual       ~LrFunction    (void) ;

    virtual double func          (double * w             ) = 0 ;
    virtual void   grad          (double * w, double * g ) = 0 ;
    virtual void   Hv            (double * s, double * Hs) = 0 ;
    virtual int    getNrVariable (void                   ) = 0 ;

  protected:

  private:

};

class Q_NLP_EXPORT LrL2R : public LrFunction
{
  public:

    explicit       LrL2R         (LrProblem * problem,double * C) ;
    explicit       LrL2R         (void) ;
    virtual       ~LrL2R         (void) ;

    virtual double func          (double * w             ) ;
    virtual void   grad          (double * w, double * g ) ;
    virtual void   Hv            (double * s, double * Hs) ;
    virtual int    getNrVariable (void                   ) ;

  protected:

    LrProblem * problem ;
    double    * C       ;
    double    * z       ;
    double    * D       ;

    void Xv  (double *v, double *Xv ) ;
    void XTv (double *v, double *XTv) ;

  private:

};

class Q_NLP_EXPORT LrL2Svc : public LrFunction
{
  public:

    explicit       LrL2Svc       (LrProblem * problem,double * C) ;
    explicit       LrL2Svc       (void) ;
    virtual       ~LrL2Svc       (void) ;

    virtual double func          (double * w             ) ;
    virtual void   grad          (double * w, double * g ) ;
    virtual void   Hv            (double * s, double * Hs) ;
    virtual int    getNrVariable (void                   ) ;

  protected:

    LrProblem * problem ;
    double    * C       ;
    double    * z       ;
    double    * D       ;
    int       * I       ;
    int         sizeI   ;

    void Xv     (double * v , double * Xv  ) ;
    void subXv  (double * v , double * Xv  ) ;
    void subXTv (double * v , double * XTv ) ;

  private:

};

class Q_NLP_EXPORT LrL2Svr : public LrL2Svc
{
  public:

    explicit LrL2Svr (LrProblem * problem,double * C,double p) ;
    virtual ~LrL2Svr (void) ;

    double   func    (double * w) ;
    void     grad    (double * w,double * g) ;

  protected:

    double p ;

  private:

};

class Q_NLP_EXPORT LrSolver
{
  public:

    explicit LrSolver (LrProblem * problem            ,
                       int         nr_class           ,
                       double    * C                  ,
                       double      eps      = 0.1     ,
                       int         max_iter = 100000) ;
    virtual ~LrSolver (void);

    void     Solve    (double * w) ;

  protected:

    int         wSize   ;
    int         l       ;
    int         nrClass ;
    int         maxIter ;
    double      eps     ;
    double    * B       ;
    double    * C       ;
    double    * G       ;
    LrProblem * problem ;

    void subSolve (double A_i,int yi,double C_yi,int active_i,double * alpha_new) ;
    bool beShrunk (int i,int m,int yi,double alpha_i,double minG) ;

  private:

};

class Q_NLP_EXPORT LrTron
{
  public:

    explicit LrTron (LrFunction * fun_obj,double eps = 0.1,int max_iter = 1000) ;
    virtual ~LrTron (void) ;

    void     tron   (double * w) ;

  protected:

    double       eps      ;
    int          max_iter ;
    LrFunction * fun_obj  ;

    int    trcg     (double delta,double * g,double * s,double * r) ;
    double normInf  (int n,double * x) ;

  private:

};

class Q_NLP_EXPORT Linear
{
  public:

    LrParameter Parameter ;
    int         NrClass   ; /* number of classes */
    int         NrFeature ;
    double    * w         ;
    int       * label     ; /* label of each class */
    double      bias      ;

    explicit Linear             (void) ;
    virtual ~Linear             (void) ;

    bool     ProbabilityModel   (void) ;
    Linear & Train              (LrProblem   & problem   ,
                                 LrParameter & param   ) ;
    void     CrossValidation    (LrProblem   & problem   ,
                                 LrParameter & param     ,
                                 int           nr_fold   ,
                                 double      * target  ) ;
    double   PredictValues      (LrNode & x,double * dec_values) ;
    double   PredictProbability (LrNode & x,double * probEstimates) ;
    double   Predict            (LrNode & x) ;

  protected:

  private:

};

}

}

Q_DECLARE_METATYPE(N::NLP::LrNode)
Q_DECLARE_METATYPE(N::NLP::LrProblem)
Q_DECLARE_METATYPE(N::NLP::LrParameter)
Q_DECLARE_METATYPE(N::NLP::LrL2R)
Q_DECLARE_METATYPE(N::NLP::LrL2Svc)
Q_DECLARE_METATYPE(N::NLP::Linear)

Q_DECLARE_INTERFACE(N::NLP::LrFunction , "com.neutrino.nlp.function" )

QT_END_NAMESPACE

#endif
