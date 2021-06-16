#include <qtnlp.h>

#ifndef SwapValue
template <class T> static inline void SwapValue(T & x, T & y) { T t=x; x=y; y=t; }
#endif

#ifndef MinValue
template <class T> static inline T MinValue(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef MaxValue
template <class T> static inline T MaxValue(T x,T y) { return (x>y)?x:y; }
#endif

#define INF HUGE_VAL

//////////////////////////////////////////////////////////////////////////////

// A coordinate descent algorithm for
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
//   upper_bound_i = Cp if y_i = 1
//   upper_bound_i = Cn if y_i = -1
//   D_ii = 0
// In L2-SVM case:
//   upper_bound_i = INF
//   D_ii = 1/(2*Cp) if y_i = 1
//   D_ii = 1/(2*Cn) if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 3 of Hsieh et al., ICML 2008

#undef GETI
#define GETI(i) (y[i]+1)

// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svc            (
              N::NLP::LrProblem * problem ,
              double            * w       ,
              double              eps     ,
              double              Cp      ,
              double              Cn      ,
              int                 solver  )
{
  int      l      = problem->l                ;
  int      w_size = problem->n                ;
  int      iter   = 0                         ;
  double   C                                  ;
  double   d                                  ;
  double   G                                  ;
  double * QD          = new double [l]       ;
  int      max_iter    = 1000                 ;
  int    * index       = new int    [l]       ;
  double * alpha       = new double [l]       ;
  char   * y           = new char   [l]       ;
  int      active_size = l                    ;
  /////////////////////////////////////////////
  // PG: projected gradient, for shrinking and stopping
  double PG                                   ;
  double PGmax_old =  INF                     ;
  double PGmin_old = -INF                     ;
  double PGmax_new                            ;
  double PGmin_new                            ;
  /////////////////////////////////////////////
  // default solver_type: L2R_L2LOSS_SVC_DUAL
  double diag       [3] = {0.5/Cn, 0, 0.5/Cp} ;
  double upper_bound[3] = {INF   , 0, INF   } ;
  /////////////////////////////////////////////
  if (solver == N::NLP::LrParameter::L2R_L1LOSS_SVC_DUAL) {
    diag        [0] = 0                       ;
    diag        [2] = 0                       ;
    upper_bound [0] = Cn                      ;
    upper_bound [2] = Cp                      ;
  }                                           ;
  problem -> Camp()                           ;
  /////////////////////////////////////////////
  // Initial alpha can be set here. Note that
  // 0 <= alpha[i] <= upper_bound[GETI(i)]
  nFullLoop ( i , l      ) alpha [i] = 0      ;
  nFullLoop ( i , w_size ) w     [i] = 0      ;
  nFullLoop ( i , l )                         {
    QD[i] = diag[GETI(i)]                     ;
    N::NLP::LrNode * xi = problem->x[i]       ;
    while ( xi->index != -1 )                 {
      double val = xi->value                  ;
      QD[i] += nSquare(val)                   ;
      w[xi->index-1] += y[i]*alpha[i]*val     ;
      xi++                                    ;
    }                                         ;
    index[i] = i                              ;
  }                                           ;
  /////////////////////////////////////////////
  while (iter < max_iter)                     {
    PGmax_new = -INF                          ;
    PGmin_new =  INF                          ;
    ///////////////////////////////////////////
    nFullLoop ( i , active_size )             {
      int j = i+rand()%(active_size-i)        ;
      SwapValue(index[i], index[j])                ;
    }                                         ;
    ///////////////////////////////////////////
    nFullLoop ( s , active_size )             {
      int  i  = index [s]                     ;
      char yi = y     [i]                     ;
      G = 0                                   ;
      /////////////////////////////////////////
      N::NLP::LrNode * xi = problem->x[i]     ;
      while ( xi->index != -1)                {
        G += w[xi->index-1] * (xi->value)     ;
        xi++                                  ;
      }                                       ;
      /////////////////////////////////////////
      G  = G * yi - 1                         ;
      C  = upper_bound     [ GETI(i) ]        ;
      G += alpha[i] * diag [ GETI(i) ]        ;
      /////////////////////////////////////////
      PG = 0                                  ;
      if (alpha[i] == 0)                      {
        if (G > PGmax_old)                    {
          active_size--                       ;
          SwapValue(index[s], index[active_size])  ;
          s--                                 ;
          continue                            ;
        } else
        if (G < 0) PG = G                     ;
      } else
      if (alpha[i] == C)                      {
        if (G < PGmin_old)                    {
          active_size--                       ;
          SwapValue(index[s], index[active_size])  ;
          s--                                 ;
          continue                            ;
        } else
        if (G > 0) PG = G                     ;
      } else PG = G                           ;
      /////////////////////////////////////////
      PGmax_new = MaxValue(PGmax_new, PG)          ;
      PGmin_new = MinValue(PGmin_new, PG)          ;
      /////////////////////////////////////////
      if ( fabs(PG) > 1.0e-12 )               {
        double alpha_old = alpha[i]           ;
        alpha[i] = MinValue                  (
                    MaxValue(alpha[i] - G/QD[i],0.0) ,
                    C                       ) ;
        d  = (alpha[i] - alpha_old) * yi      ;
        xi = problem->x[i]                    ;
        while ( xi->index != -1 )             {
          w[xi->index-1] += d * xi->value     ;
          xi++                                ;
        }                                     ;
      }                                       ;
    }                                         ;
    ///////////////////////////////////////////
    iter++                                    ;
    ///////////////////////////////////////////
    if ( PGmax_new - PGmin_new <= eps )       {
      if (active_size == l) break ; else      {
        active_size =  l                      ;
        PGmax_old   =  INF                    ;
        PGmin_old   = -INF                    ;
        continue                              ;
      }                                       ;
    }                                         ;
    ///////////////////////////////////////////
    PGmax_old = PGmax_new                     ;
    PGmin_old = PGmin_new                     ;
    if (PGmax_old <= 0) PGmax_old =  INF      ;
    if (PGmin_old >= 0) PGmin_old = -INF      ;
  }                                           ;
  /////////////////////////////////////////////
  // calculate objective value
  double v   = 0                              ;
  int    nSV = 0                              ;
  nFullLoop ( i , w_size ) v += nSquare(w[i]) ;
  nFullLoop ( i , l )                         {
    v += alpha[i]*(alpha[i]*diag[GETI(i)]-2)  ;
    if ( alpha[i] > 0 ) ++nSV                 ;
  }                                           ;
  /////////////////////////////////////////////
  nDeleteArray ( QD    )                      ;
  nDeleteArray ( alpha )                      ;
  nDeleteArray ( y     )                      ;
  nDeleteArray ( index )                      ;
}

//////////////////////////////////////////////////////////////////////////////

// A coordinate descent algorithm for
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
//
//  where Qij = xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
//   upper_bound_i = C
//   lambda_i = 0
// In L2-SVM case:
//   upper_bound_i = INF
//   lambda_i = 1/(2*C)
//
// Given:
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012

#undef GETI
#define GETI(i) (0)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svr              (
              N::NLP::LrProblem   * problem ,
              double              * w       ,
              N::NLP::LrParameter * param   ,
              int                   solver  )
{
  int      l           = problem->l                     ;
  int      w_size      = problem->n                     ;
  int      iter        = 0                              ;
  int      max_iter    = 1000                           ;
  int      active_size = l                              ;
  int    * index       = new int[l]                     ;
  double   C           = param->C                       ;
  double   p           = param->p                       ;
  double   eps         = param->eps                     ;
  double   Gmax_old    = INF                            ;
  double   Gmax_new                                     ;
  double   Gnorm1_new                                   ;
  double   Gnorm1_init                                  ;
  double   d                                            ;
  double   G                                            ;
  double   H                                            ;
  double * beta        = new double[l]                  ;
  double * QD          = new double[l]                  ;
  double * y           = problem->y                     ;
  double   lambda      [1]                              ; // L2R_L2LOSS_SVR_DUAL
  double   upper_bound [1]                              ;
  lambda      [ 0 ] = 0.5 / C                           ;
  upper_bound [ 0 ] = INF                               ;
  ///////////////////////////////////////////////////////
  if ( solver == N::NLP::LrParameter::L2R_L1LOSS_SVR_DUAL ) {
    lambda      [ 0 ] = 0                               ;
    upper_bound [ 0 ] = C                               ;
  }                                                     ;
  // Initial beta can be set here. Note that
  // -upper_bound <= beta[i] <= upper_bound
  nFullLoop ( i , l      ) beta [ i ] = 0               ;
  nFullLoop ( i , w_size ) w    [ i ] = 0               ;
  nFullLoop ( i , l )                                   {
    N::NLP::LrNode * xi = problem->x[i]                 ;
    QD[i] = 0                                           ;
    while ( xi->index != -1 )                           {
      double val = xi->value                            ;
      QD[ i           ] += nSquare(val)                 ;
      w [ xi->index-1 ] += beta[i] * val                ;
      xi++                                              ;
    }                                                   ;
    index[i] = i                                        ;
  }                                                     ;
  ///////////////////////////////////////////////////////
  while ( iter < max_iter )                             {
    Gmax_new   = 0                                      ;
    Gnorm1_new = 0                                      ;
    nFullLoop(i,active_size)                            {
      int j = i + rand()%(active_size-i)                ;
      SwapValue(index[i],index[j])                           ;
    }                                                   ;
    nFullLoop ( s , active_size )                       {
      int i = index[s]                                  ;
      G = -y[i] + lambda[GETI(i)] * beta[i]             ;
      H = QD[i] + lambda[GETI(i)]                       ;
      ///////////////////////////////////////////////////
      N::NLP::LrNode * xi = problem->x[i]               ;
      while ( xi->index != -1 )                         {
        int    ind = xi->index - 1                      ;
        double val = xi->value                          ;
        G         += val * w[ind]                       ;
        xi++                                            ;
      }                                                 ;
      ///////////////////////////////////////////////////
      double Gp        = G + p                          ;
      double Gn        = G - p                          ;
      double violation = 0                              ;
      ///////////////////////////////////////////////////
      if ( beta[i] == 0 )                               {
        if ( Gp < 0 ) violation = -Gp ;              else
        if ( Gn > 0 ) violation =  Gn ;              else
        if ( Gp > Gmax_old && Gn < -Gmax_old )          {
          active_size --                                ;
          SwapValue ( index[s] , index[active_size] )        ;
          s--                                           ;
          continue                                      ;
        }                                               ;
      } else
      if ( beta[i] >= upper_bound[GETI(i)] )            {
        if ( Gp > 0 ) violation = Gp ;               else
        if ( Gp < -Gmax_old )                           {
          active_size--                                 ;
          SwapValue ( index[s] , index[active_size] )        ;
          s--                                           ;
          continue                                      ;
        }                                               ;
      } else
      if ( beta[i] <= -upper_bound[GETI(i)] )           {
        if ( Gn < 0 ) violation = -Gn ;              else
        if ( Gn > Gmax_old )                            {
          active_size--                                 ;
          SwapValue ( index[s] , index[active_size] )        ;
          s--                                           ;
          continue                                      ;
        }                                               ;
      } else
      if (beta[i] > 0) violation = fabs ( Gp )          ;
                  else violation = fabs ( Gn )          ;
      ///////////////////////////////////////////////////
      Gmax_new    = MaxValue ( Gmax_new , violation )        ;
      Gnorm1_new += violation                           ;
      // obtain Newton direction d
      if ( Gp < ( H * beta[i] ) ) d = -Gp / H  ;     else
      if ( Gn > ( H * beta[i] ) ) d = -Gn / H  ;     else
                                  d = -beta[i]          ;
      if ( fabs(d) < 1.0e-12 ) continue                 ;
      ///////////////////////////////////////////////////
      double beta_old = beta[i]                         ;
      beta[i] = MinValue ( MaxValue ( beta[i]+d                   ,
                            -upper_bound[GETI(i)]     ) ,
                      upper_bound[GETI(i)]            ) ;
      d = beta[i] - beta_old                            ;
      if ( d != 0 )                                     {
        xi = problem->x[i]                              ;
        while ( xi->index != -1 )                       {
          w [ xi->index - 1 ] += d * xi->value          ;
          xi++                                          ;
        }                                               ;
      }                                                 ;
    }                                                   ;
    /////////////////////////////////////////////////////
    if (iter == 0) Gnorm1_init = Gnorm1_new             ;
    iter++                                              ;
    if ( Gnorm1_new <= (eps * Gnorm1_init) )            {
      if ( active_size == l ) break; else               {
        active_size = l                                 ;
        Gmax_old    = INF                               ;
        continue                                        ;
      }                                                 ;
    }                                                   ;
    Gmax_old = Gmax_new                                 ;
  }                                                     ;
  ///////////////////////////////////////////////////////
  // calculate objective value
  double v   = 0                                        ;
  int    nSV = 0                                        ;
  nFullLoop ( i , w_size ) v += nSquare(w[i])           ;
  v *= 0.5                                              ;
  nFullLoop ( i , l )                                   {
    v += p    * fabs(beta[i])                           -
         y[i] * beta[i]                                 +
         0.5  * lambda[GETI(i)] * nSquare(beta[i])      ;
    if ( beta[i] != 0 )  nSV++                          ;
  }                                                     ;
  ///////////////////////////////////////////////////////
  nDeleteArray ( beta  )                                ;
  nDeleteArray ( QD    )                                ;
  nDeleteArray ( index )                                ;
}

//////////////////////////////////////////////////////////////////////////////

// A coordinate descent algorithm for
// the dual of L2-regularized logistic regression problems
//
//  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i),
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  upper_bound_i = Cp if y_i = 1
//  upper_bound_i = Cn if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 5 of Yu et al., MLJ 2010

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void solve_l2r_lr_dual             (
       N::NLP::LrProblem * problem ,
       double            * w       ,
       double              eps     ,
       double              Cp      ,
       double              Cn      )
{
  int      l              = problem->l                      ;
  int      w_size         = problem->n                      ;
  int      iter           = 0                               ;
  int      max_iter       = 1000                            ;
  int      max_inner_iter = 100                             ; // for inner Newton
  int    * index          = new int    [ l     ]            ;
  double * xTx            = new double [ l     ]            ;
  double * alpha          = new double [ l * 2 ]            ; // store alpha and C - alpha
  char   * y              = new char   [ l     ]            ;
  double   innereps       = 1e-2                            ;
  double   innereps_min   = MinValue ( 1e-8 , eps )              ;
  double   upper_bound[3] = { Cn , 0 , Cp }                 ;
  ///////////////////////////////////////////////////////////
  problem -> Camp()                                         ;
  ///////////////////////////////////////////////////////////
  // Initial alpha can be set here. Note that
  // 0 < alpha[i] < upper_bound[GETI(i)]
  // alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
  nFullLoop ( i , l )                                       {
    alpha[2*i  ] = MinValue(0.001*upper_bound[GETI(i)], 1e-8)    ;
    alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i]        ;
  }                                                         ;
  ///////////////////////////////////////////////////////////
  nFullLoop ( i , w_size ) w[i] = 0                         ;
  nFullLoop ( i , l )                                       {
    xTx[i] = 0                                              ;
    N::NLP::LrNode * xi = problem->x[i]                     ;
    while ( xi->index != -1 )                               {
      double val = xi->value                                ;
      xTx[i]         += nSquare(val)                        ;
      w[xi->index-1] += y[i] * alpha[2*i] * val             ;
      xi++                                                  ;
    }                                                       ;
    index[i] = i                                            ;
  }                                                         ;
  ///////////////////////////////////////////////////////////
  while (iter < max_iter)                                   {
    int    newton_iter = 0                                  ;
    double Gmax        = 0                                  ;
    /////////////////////////////////////////////////////////
    nFullLoop ( i , l )                                     {
      int j = i + rand()%(l-i)                              ;
      SwapValue ( index[i] , index[j] )                          ;
    }                                                       ;
    /////////////////////////////////////////////////////////
    nFullLoop ( s , l )                                     {
      int          i    = index       [ s     ]             ;
      char         yi   = y           [ i     ]             ;
      double       C    = upper_bound [GETI(i)]             ;
      double       ywTx = 0                                 ;
      double       xisq = xTx         [ i     ]             ;
      N::NLP::LrNode * xi   = problem->x  [ i     ]         ;
      ///////////////////////////////////////////////////////
      while ( xi->index != -1 )                             {
        ywTx += w[ xi->index - 1 ] * xi->value              ;
        xi++                                                ;
      }                                                     ;
      ywTx *= y[i]                                          ;
      ///////////////////////////////////////////////////////
      double a = xisq                                       ;
      double b = ywTx                                       ;
      ///////////////////////////////////////////////////////
      // Decide to minimize g_1(z) or g_2(z)
      int ind1 = 2 * i                                      ;
      int ind2 = 2 * i + 1                                  ;
      int sign = 1                                          ;
      if ( 0.5 * a * ( alpha[ind2] - alpha[ind1] ) + b < 0) {
        ind1 =  2 * i + 1                                   ;
        ind2 =  2 * i                                       ;
        sign = -1                                           ;
      }                                                     ;
      ///////////////////////////////////////////////////////
      //  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
      double alpha_old = alpha[ind1]                        ;
      double z         = alpha_old                          ;
      if ( C - z < 0.5 * C ) z *= 0.1                       ;
      double gp = a * ( z - alpha_old )                     +
                  b * sign                                  +
                  log ( z / (C-z) )                         ;
      Gmax = MaxValue ( Gmax , fabs(gp) )                        ;
      ///////////////////////////////////////////////////////
      // Newton method on the sub-problem
      const double eta        = 0.1                         ; // xi in the paper
      int          inner_iter = 0                           ;
      while ( inner_iter <= max_inner_iter )                {
        if ( fabs(gp) < innereps ) break                    ;
        double gpp  = a + C  / (C-z) / z                    ;
        double tmpz = z - gp / gpp                          ;
        if ( tmpz <= 0 ) z *= eta                           ;
                    else z  = tmpz                          ; // tmpz in (0, C)
        gp = a*(z-alpha_old)+sign*b+log(z/(C-z))            ;
        newton_iter ++                                      ;
        inner_iter  ++                                      ;
      }                                                     ;
      if (inner_iter > 0)                                   { // update w
        alpha[ind1] = z                                     ;
        alpha[ind2] = C - z                                 ;
        xi          = problem->x[i]                         ;
        while ( xi->index != -1 )                           {
          w[xi->index-1] += sign * (z-alpha_old)            *
                            yi   * xi->value                ;
          xi++                                              ;
        }                                                   ;
      }                                                     ;
    }                                                       ;
    /////////////////////////////////////////////////////////
    iter++                                                  ;
    if ( Gmax < eps ) break                                 ;
    if ( newton_iter <= l/10 )                              {
      innereps = MaxValue(innereps_min,0.1*innereps)             ;
    }                                                       ;
  }                                                         ;
  ///////////////////////////////////////////////////////////
  // calculate objective value
  double v = 0                                              ;
  nFullLoop ( i , w_size ) v += nSquare(w[i])               ;
  v *= 0.5                                                  ;
  nFullLoop ( i , l )                                       {
    v += alpha[2*i  ] * log(alpha[2*i  ])                   +
         alpha[2*i+1] * log(alpha[2*i+1])                   -
         upper_bound[GETI(i)] * log(upper_bound[GETI(i)])   ;
  }                                                         ;
  ///////////////////////////////////////////////////////////
  nDeleteArray ( xTx   )                                    ;
  nDeleteArray ( alpha )                                    ;
  nDeleteArray ( y     )                                    ;
  nDeleteArray ( index )                                    ;
}

//////////////////////////////////////////////////////////////////////////////

// A coordinate descent algorithm for
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_l2_svc               (
              N::NLP::LrProblem * prob_col ,
              double            * w        ,
              double              eps      ,
              double              Cp       ,
              double              Cn       )
{
  int          l                  = prob_col->l ;
  int          w_size             = prob_col->n ;
  int          iter               = 0           ;
  int          max_iter           = 1000        ;
  int          active_size        = w_size      ;
  int          max_num_linesearch = 20          ;
  double       sigma              = 0.01        ;
  double       Gmax_old           = INF         ;
  double       Gmax_new                         ;
  double       Gnorm1_new                       ;
  double       Gnorm1_init                      ;
  double       d                                ;
  double       G_loss                           ;
  double       G                                ;
  double       H                                ;
  double       d_old                            ;
  double       d_diff                           ;
  double       loss_old                         ;
  double       loss_new                         ;
  double       appxcond                         ;
  double       cond                             ;
  int        * index = new int    [ w_size ]    ;
  char       * y     = new char   [ l      ]    ;
  double     * b     = new double [ l      ]    ; // b = 1-ywTx
  double     * xj_sq = new double [ w_size ]    ;
  N::NLP::LrNode * x                            ;
  double       C[3]  = { Cn , 0 , Cp }          ;
  ///////////////////////////////////////////////
  prob_col -> Camp ( )                          ;
  nFullLoop ( i , l      ) b[i] = 1             ;
  nFullLoop ( i , w_size ) w[i] = 0             ;
  nFullLoop ( i , w_size )                      {
    index[i] = i                                ;
    xj_sq[i] = 0                                ;
    x = prob_col->x[i]                          ;
    while ( x->index != -1 )                    {
      int ind     = x->index - 1                ;
      x->value   *= y[ind]                      ; // x->value stores yi*xij
      double val  = x->value                    ;
      b    [ind] -= w[i] * val                  ;
      xj_sq[i  ] += C[GETI(ind)] * nSquare(val) ;
      x++                                       ;
    }                                           ;
  }                                             ;
  ///////////////////////////////////////////////
  while ( iter < max_iter )                     {
    Gmax_new   = 0                              ;
    Gnorm1_new = 0                              ;
    /////////////////////////////////////////////
    nFullLoop ( j , active_size )               {
      int i = j + rand() % (active_size-j)      ;
      SwapValue(index[i],index[j])                   ;
    }                                           ;
    /////////////////////////////////////////////
    nFullLoop ( s , active_size )               {
      int j  = index[s]                         ;
      G_loss = 0                                ;
      H      = 0                                ;
      ///////////////////////////////////////////
      x = prob_col->x[j]                        ;
      while ( x->index != -1 )                  {
        int ind = x->index - 1                  ;
        if ( b[ind] > 0 )                       {
          double val = x->value                 ;
          double tmp = C[GETI(ind)] * val       ;
          G_loss    -= tmp * b[ind]             ;
          H         += tmp * val                ;
        }                                       ;
        x++                                     ;
      }                                         ;
      ///////////////////////////////////////////
      G_loss *= 2                               ;
      G       = G_loss                          ;
      H      *= 2                               ;
      H       = MaxValue(H, 1e-12)                   ;
      ///////////////////////////////////////////
      double Gp        = G + 1                  ;
      double Gn        = G - 1                  ;
      double violation = 0                      ;
      ///////////////////////////////////////////
      if ( w[j] == 0 )                          {
        if ( Gp < 0 ) violation = -Gp      ; else
        if ( Gn > 0 ) violation =  Gn      ; else
        if ( Gp>Gmax_old/l && Gn<-Gmax_old/l )  {
          active_size--                         ;
          SwapValue(index[s],index[active_size])     ;
          s--                                   ;
          continue                              ;
        }                                       ;
      } else
      if(w[j] > 0) violation = fabs(Gp)         ;
              else violation = fabs(Gn)         ;
      ///////////////////////////////////////////
      Gmax_new    = MaxValue(Gmax_new,violation)     ;
      Gnorm1_new += violation                   ;
      ///////////////////////////////////////////
      // obtain Newton direction d
      if ( Gp < ( H * w[j] ) ) d = -Gp/H ;   else
      if ( Gn > ( H * w[j] ) ) d = -Gn/H ;   else
                               d = -w[j]        ;
      if ( fabs(d) < 1.0e-12 ) continue         ;
      ///////////////////////////////////////////
      double delta                              ;
      int    num_linesearch                     ;
      delta = fabs(w[j]+d) - fabs(w[j]) + (G*d) ;
      d_old = 0                                 ;
      ///////////////////////////////////////////
      for (num_linesearch=0                     ;
           num_linesearch < max_num_linesearch  ;
           num_linesearch++                   ) {
        d_diff   = d_old - d                    ;
        cond     = fabs(w[j]+d)                 -
                   fabs(w[j]  )                 -
                   (sigma*delta)                ;
        appxcond = xj_sq[j] * nSquare(d)        +
                   G_loss   * d                 +
                   cond                         ;
        /////////////////////////////////////////
        if ( appxcond <= 0 )                    {
          x = prob_col->x[j]                    ;
          while ( x->index != -1 )              {
            b[x->index-1] += d_diff * x->value  ;
            x++                                 ;
          }                                     ;
          break                                 ;
        }                                       ;
        /////////////////////////////////////////
        if ( num_linesearch == 0 )              {
          loss_old = 0                          ;
          loss_new = 0                          ;
          x        = prob_col->x[j]             ;
          while ( x->index != -1 )              {
            int ind = x->index-1                ;
            if (b[ind] > 0)                     {
              double bv = b[ind]                ;
              bv = nSquare(bv)                  ;
              loss_old += C[ GETI(ind) ] * bv   ;
            }                                   ;
            /////////////////////////////////////
            double b_new = b[ind]               +
                           d_diff * x->value    ;
            b[ind] = b_new                      ;
            if (b_new > 0)                      {
              double bv = nSquare(b_new)        ;
              loss_new += C[GETI(ind)] * bv     ;
            }                                   ;
            x++                                 ;
          }                                     ;
        } else                                  {
          loss_new = 0                          ;
          x        = prob_col->x[j]             ;
          while ( x->index != -1 )              {
            int    ind   = x->index - 1         ;
            double b_new = b[ind]               +
                           d_diff * x->value    ;
            b[ind] = b_new                      ;
            if (b_new > 0)                      {
              double bv = b_new                 ;
              bv = nSquare(bv)                  ;
              loss_new += C[GETI(ind)] * bv     ;
            }                                   ;
            x++                                 ;
          }                                     ;
        }                                       ;
        /////////////////////////////////////////
        cond = cond + loss_new - loss_old       ;
        if ( cond <= 0 ) break ; else           {
          d_old  = d                            ;
          d     *= 0.5                          ;
          delta *= 0.5                          ;
        }                                       ;
      }                                         ;
      ///////////////////////////////////////////
      w[j] += d                                 ;
      // recompute b[] if line search takes too many steps
      if (num_linesearch >= max_num_linesearch) {
        nFullLoop ( i , l ) b[i] = 1            ;
        nFullLoop ( i , w_size )                {
          if ( w[i] == 0 ) continue             ;
          x = prob_col->x[i]                    ;
          while ( x->index != -1 )              {
            b [ x->index-1 ] -= w[i] * x->value ;
            x++                                 ;
          }                                     ;
        }                                       ;
      }                                         ;
    }                                           ;
    /////////////////////////////////////////////
    if ( iter == 0 ) Gnorm1_init = Gnorm1_new   ;
    iter++                                      ;
    /////////////////////////////////////////////
    if ( Gnorm1_new <= ( eps * Gnorm1_init ) )  {
      if ( active_size == w_size ) break ; else {
        active_size = w_size                    ;
        Gmax_old    = INF                       ;
        continue                                ;
      }                                         ;
    }                                           ;
    Gmax_old = Gmax_new                         ;
  }                                             ;
  ///////////////////////////////////////////////
  // calculate objective value
  double v   = 0                                ;
  int    nnz = 0                                ;
  nFullLoop ( j , w_size )                      {
    x = prob_col->x[j]                          ;
    while ( x->index != -1 )                    {
      x->value *= prob_col->y[x->index-1]       ; // restore x->value
      x++                                       ;
    }                                           ;
    if ( w[j] != 0 )                            {
      v  += fabs(w[j])                          ;
      nnz++                                     ;
    }                                           ;
  }                                             ;
  ///////////////////////////////////////////////
  nFullLoop ( j , l)                            {
    if (b[j] > 0)                               {
      double bv = b[j]                          ;
      v += C[GETI(j)] * nSquare(bv)             ;
    }                                           ;
  }                                             ;
  ///////////////////////////////////////////////
  nDeleteArray ( index )                        ;
  nDeleteArray ( y     )                        ;
  nDeleteArray ( b     )                        ;
  nDeleteArray ( xj_sq )                        ;
}

//////////////////////////////////////////////////////////////////////////////

// A coordinate descent algorithm for
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_lr                   (
              N::NLP::LrProblem * prob_col ,
              double            * w        ,
              double              eps      ,
              double              Cp       ,
              double              Cn       )
{
  int          l                  = prob_col->l ;
  int          w_size             = prob_col->n ;
  int          newton_iter        = 0           ;
  int          iter               = 0           ;
  int          max_newton_iter    = 100         ;
  int          max_iter           = 1000        ;
  int          max_num_linesearch = 20          ;
  int          active_size                      ;
  int          QP_active_size                   ;
  double       nu                 = 1e-12       ;
  double       inner_eps          = 1           ;
  double       sigma              = 0.01        ;
  double       w_norm                           ;
  double       w_norm_new                       ;
  double       z                                ;
  double       G                                ;
  double       H                                ;
  double       Gnorm1_init                      ;
  double       Gmax_old           = INF         ;
  double       Gmax_new                         ;
  double       Gnorm1_new                       ;
  double       QP_Gmax_old        = INF         ;
  double       QP_Gmax_new                      ;
  double       QP_Gnorm1_new                    ;
  double       delta                            ;
  double       negsum_xTd                       ;
  double       cond                             ;
  int        * index       = new int   [w_size] ;
  char       * y           = new char  [l     ] ;
  double     * Hdiag       = new double[w_size] ;
  double     * Grad        = new double[w_size] ;
  double     * wpd         = new double[w_size] ;
  double     * xjneg_sum   = new double[w_size] ;
  double     * xTd         = new double[l     ] ;
  double     * exp_wTx     = new double[l     ] ;
  double     * exp_wTx_new = new double[l     ] ;
  double     * tau         = new double[l     ] ;
  double     * D           = new double[l     ] ;
  N::NLP::LrNode * x                            ;
  double       C [ 3 ]     = { Cn , 0 , Cp }    ;
  ///////////////////////////////////////////////
  // Initial w can be set here.
  nFullLoop ( i , w_size ) w       [ i ] = 0    ;
  nFullLoop ( i , l      ) exp_wTx [ i ] = 0    ;
  prob_col -> Camp ( )                          ;
  w_norm = 0                                    ;
  ///////////////////////////////////////////////
  nFullLoop ( j , w_size )                      {
    w_norm        += fabs(w[j])                 ;
    wpd       [j]  = w[j]                       ;
    index     [j]  = j                          ;
    xjneg_sum [j]  = 0                          ;
    x              = prob_col->x[j]             ;
    while ( x->index != -1 )                    {
      int    ind    = x->index - 1              ;
      double val    = x->value                  ;
      exp_wTx[ind] += ( w[j] * val )            ;
      if ( y[ind] == -1 )                       {
        xjneg_sum[j] += C[GETI(ind)] * val      ;
      }                                         ;
      x++                                       ;
    }                                           ;
  }                                             ;
  ///////////////////////////////////////////////
  nFullLoop ( j , l )                           {
    exp_wTx[j] = exp(exp_wTx[j])                ;
    double tau_tmp = 1/(1+exp_wTx[j])           ;
    tau [j] = C[GETI(j)] * tau_tmp              ;
    D   [j] = tau[j] * exp_wTx[j] * tau_tmp     ;
  }                                             ;
  ///////////////////////////////////////////////
  while ( newton_iter < max_newton_iter )       {
    Gmax_new    = 0                             ;
    Gnorm1_new  = 0                             ;
    active_size = w_size                        ;
    nFullLoop ( s , active_size )               {
      int j     = index[s]                      ;
      Hdiag [j] = nu                            ;
      Grad  [j] = 0                             ;
      ///////////////////////////////////////////
      double tmp = 0                            ;
      x = prob_col->x[j]                        ;
      ///////////////////////////////////////////
      while ( x->index != -1 )                  {
        int ind   = x->index - 1                ;
        Hdiag[j] += x->value * x->value*D[ind]  ;
        tmp      += x->value * tau[ind]         ;
        x++                                     ;
      }                                         ;
      ///////////////////////////////////////////
      Grad[j] = -tmp + xjneg_sum[j]             ;
      ///////////////////////////////////////////
      double Gp        = Grad[j] + 1            ;
      double Gn        = Grad[j] - 1            ;
      double violation = 0                      ;
      ///////////////////////////////////////////
      if ( w[j] == 0 )                          {
        if ( Gp < 0 ) violation = -Gp      ; else
        if ( Gn > 0 ) violation =  Gn      ; else
        //outer-level shrinking
        if ( Gp>Gmax_old / l && Gn<-Gmax_old/l) {
          active_size--                         ;
          SwapValue ( index[s], index[active_size] ) ;
          s--                                   ;
          continue                              ;
        }                                       ;
      } else
      if ( w[j] > 0 ) violation = fabs(Gp)      ;
                 else violation = fabs(Gn)      ;
       Gmax_new    = MaxValue(Gmax_new, violation)   ;
       Gnorm1_new += violation                  ;
    }                                           ;
    /////////////////////////////////////////////
    if ( newton_iter == 0 )                     {
      Gnorm1_init = Gnorm1_new                  ;
    }                                           ;
    if ( Gnorm1_new <= eps*Gnorm1_init ) break  ;
    /////////////////////////////////////////////
    iter           = 0                          ;
    QP_Gmax_old    = INF                        ;
    QP_active_size = active_size                ;
    nFullLoop ( i , l ) xTd[i] = 0              ;
    /////////////////////////////////////////////
    // optimize QP over wpd
    while ( iter < max_iter )                   {
      QP_Gmax_new   = 0                         ;
      QP_Gnorm1_new = 0                         ;
      nFullLoop ( j , QP_active_size )          {
        int z = j + rand() % (QP_active_size-j) ;
        SwapValue ( index[z] , index[j] )            ;
      }                                         ;
      ///////////////////////////////////////////
      nFullLoop ( s , QP_active_size )          {
        int j = index [s]                       ;
        H = Hdiag [j]                           ;
        x = prob_col->x[j]                      ;
        G = Grad[j] + (wpd[j]-w[j]) * nu        ;
        /////////////////////////////////////////
        while ( x->index != -1 )                {
          int ind = x->index-1                  ;
          G      += x->value*D[ind]*xTd[ind]    ;
          x      ++                             ;
        }                                       ;
        /////////////////////////////////////////
        double Gp        = G + 1                ;
        double Gn        = G - 1                ;
        double violation = 0                    ;
        /////////////////////////////////////////
        if ( wpd[j] == 0 )                      {
          if ( Gp < 0 ) violation = -Gp ;    else
          if ( Gn > 0 ) violation =  Gn ;    else
          //inner-level shrinking
          if (Gp >  QP_Gmax_old/l              &&
              Gn < -QP_Gmax_old/l             ) {
            QP_active_size--                    ;
            SwapValue(index[s], index[QP_active_size]);
            s--                                 ;
            continue                            ;
          }                                     ;
        } else
        if ( wpd[j] > 0 ) violation = fabs(Gp)  ;
                     else violation = fabs(Gn)  ;
        /////////////////////////////////////////
        QP_Gmax_new=MaxValue(QP_Gmax_new, violation) ;
        QP_Gnorm1_new += violation              ;
        /////////////////////////////////////////
        // obtain solution of one-variable problem
        if ( Gp < H * wpd[j] ) z = -Gp/H ;   else
        if ( Gn > H * wpd[j] ) z = -Gn/H ;   else
                               z = -wpd[j]      ;
        /////////////////////////////////////////
        if ( fabs(z) < 1.0e-12 ) continue       ;
        z       = MinValue(MaxValue(z,-10.0),10.0)        ;
        wpd[j] += z                             ;
        x       = prob_col->x[j]                ;
        while ( x->index != -1 )                {
          int ind   = x->index - 1              ;
          xTd[ind] += x->value * z              ;
          x++                                   ;
        }                                       ;
      }                                         ;
      ///////////////////////////////////////////
      iter++                                    ;
      if (QP_Gnorm1_new<=inner_eps*Gnorm1_init) {
        //inner stopping
        if (QP_active_size==active_size) break  ;
        //active set reactivation
        else                                    {
          QP_active_size = active_size          ;
          QP_Gmax_old    = INF                  ;
          continue                              ;
        }                                       ;
      }                                         ;
      ///////////////////////////////////////////
      QP_Gmax_old = QP_Gmax_new                 ;
    }                                           ;
    /////////////////////////////////////////////
    delta      = 0                              ;
    w_norm_new = 0                              ;
    nFullLoop ( j , w_size )                    {
      delta += Grad[j] * ( wpd[j] - w[j] )      ;
      if ( wpd[j] != 0 )                        {
        w_norm_new += fabs(wpd[j])              ;
      }                                         ;
    }                                           ;
    /////////////////////////////////////////////
    delta      += ( w_norm_new - w_norm )       ;
    negsum_xTd  = 0                             ;
    nFullLoop ( i , l)                          {
      if ( y[i] == -1 )                         {
        negsum_xTd += C[GETI(i)] * xTd[i]       ;
      }                                         ;
    }                                           ;
    /////////////////////////////////////////////
    int num_linesearch                          ;
    for(num_linesearch = 0                      ;
        num_linesearch < max_num_linesearch     ;
        num_linesearch++                      ) {
      cond = w_norm_new                         -
             w_norm                             +
             negsum_xTd                         -
             sigma * delta                      ;
      ///////////////////////////////////////////
      nFullLoop ( i , l )                       {
        double exp_xTd = exp(xTd[i])            ;
        exp_wTx_new[i] = exp_wTx[i]*exp_xTd     ;
        cond += C[GETI(i)]                      *
                log((1+exp_wTx_new[i])          /
                    (exp_xTd+exp_wTx_new[i])  ) ;
      }                                         ;
      ///////////////////////////////////////////
      if ( cond <= 0 )                          {
        w_norm = w_norm_new                     ;
        nFullLoop ( j , w_size ) w[j] = wpd[j]  ;
        nFullLoop ( i , l )                     {
          exp_wTx[i] = exp_wTx_new[i]           ;
          double tau_tmp = 1 / (1+exp_wTx[i])   ;
          tau[i] = C[GETI(i)]*tau_tmp           ;
          D  [i] = tau[i]*exp_wTx[i]*tau_tmp    ;
        }                                       ;
        break                                   ;
      } else                                    {
        w_norm_new = 0                          ;
        nFullLoop ( j , w_size )                {
          wpd[j] = (w[j]+wpd[j]) * 0.5          ;
          if ( wpd[j] != 0 )                    {
            w_norm_new += fabs(wpd[j])          ;
          }                                     ;
        }                                       ;
        delta      *= 0.5                       ;
        negsum_xTd *= 0.5                       ;
        nFullLoop ( i , l ) xTd[i] *= 0.5       ;
      }                                         ;
    }                                           ;
    /////////////////////////////////////////////
    // Recompute some info due to too many line search steps
    if (num_linesearch >= max_num_linesearch)   {
      nFullLoop ( i , l      ) exp_wTx[i] = 0   ;
      nFullLoop ( i , w_size )                  {
        if ( w[i] == 0 ) continue               ;
        x = prob_col->x[i]                      ;
        while ( x->index != -1 )                {
          exp_wTx[x->index-1] += w[i]*x->value  ;
          x++                                   ;
        }                                       ;
      }                                         ;
      nFullLoop ( i , l )                       {
        exp_wTx[i] = exp(exp_wTx[i])            ;
      }                                         ;
    }                                           ;
    if ( iter == 1 ) inner_eps *= 0.25          ;
    newton_iter ++                              ;
    Gmax_old     = Gmax_new                     ;
  }                                             ;
  ///////////////////////////////////////////////
  // calculate objective value
  double v   = 0                                ;
  int    nnz = 0                                ;
  ///////////////////////////////////////////////
  nFullLoop ( j , w_size )                      {
    if ( w[j] != 0 )                            {
      v   += fabs(w[j])                         ;
      nnz ++                                    ;
    }                                           ;
  }                                             ;
  ///////////////////////////////////////////////
  nFullLoop ( j , l )                           {
    if ( y[j] == 1 )                            {
      v += C[GETI(j)] * log(1 + 1/exp_wTx[j])   ;
    } else                                      {
      v += C[GETI(j)] * log(1 +   exp_wTx[j])   ;
    }                                           ;
  }                                             ;
  ///////////////////////////////////////////////
  nDeleteArray ( index       )                  ;
  nDeleteArray ( y           )                  ;
  nDeleteArray ( Hdiag       )                  ;
  nDeleteArray ( Grad        )                  ;
  nDeleteArray ( wpd         )                  ;
  nDeleteArray ( xjneg_sum   )                  ;
  nDeleteArray ( xTd         )                  ;
  nDeleteArray ( exp_wTx     )                  ;
  nDeleteArray ( exp_wTx_new )                  ;
  nDeleteArray ( tau         )                  ;
  nDeleteArray ( D           )                  ;
}

//////////////////////////////////////////////////////////////////////////////

// transpose matrix X from row format to column format
static void transpose                      (
              N::NLP::LrProblem *  problem     ,
              N::NLP::LrNode    ** x_space_ret ,
              N::NLP::LrProblem *  prob_col    )
{
  int          l       = problem->l          ;
  int          n       = problem->n          ;
  int          nnz     = 0                   ;
  int        * col_ptr = new int[n+1]        ;
  N::NLP::LrNode * x_space = NULL            ;
  ////////////////////////////////////////////
  prob_col->l = l                            ;
  prob_col->n = n                            ;
  prob_col->y = new double     [ l ]         ;
  prob_col->x = new N::NLP::LrNode*[ n ]     ;
  ////////////////////////////////////////////
  nFullLoop ( i , l )                        {
    prob_col->y[i] = problem->y[i]           ;
  }                                          ;
  nFullLoop ( i , (n+1) )                    {
    col_ptr[i] = 0                           ;
  }                                          ;
  ////////////////////////////////////////////
  nFullLoop ( i , l )                        {
    N::NLP::LrNode * x = problem->x[i]       ;
    while ( x->index != -1 )                 {
      nnz                  ++                ;
      col_ptr [ x->index ] ++                ;
      x++                                    ;
    }                                        ;
  }                                          ;
  ////////////////////////////////////////////
  for (int i=1;i<n+1;i++)                    {
    col_ptr[i] += col_ptr[i-1] + 1           ;
  }                                          ;
  ////////////////////////////////////////////
  x_space = new N::NLP::LrNode [ nnz+n ]     ;
  nFullLoop ( i , n )                        {
    prob_col->x[i] = &x_space [ col_ptr[i] ] ;
  }                                          ;
  ////////////////////////////////////////////
  nFullLoop ( i , l )                        {
    N::NLP::LrNode * x = problem->x[i]       ;
    while ( x->index != -1 )                 {
      int ind = x->index-1                   ;
      x_space[col_ptr[ind]].index = i+1      ; // starts from 1
      x_space[col_ptr[ind]].value = x->value ;
      col_ptr[ind]++                         ;
      x++                                    ;
    }                                        ;
  }                                          ;
  ////////////////////////////////////////////
  nFullLoop ( i , n )                        {
    x_space[col_ptr[i]].index = -1           ;
  }                                          ;
  *x_space_ret = x_space                     ;
  delete [] col_ptr                          ;
}

//////////////////////////////////////////////////////////////////////////////

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void groupClasses                        (
              N::NLP::LrProblem *  problem      ,
              int               *  nr_class_ret ,
              int               ** label_ret    ,
              int               ** start_ret    ,
              int               ** count_ret    ,
              int               *  perm         )
{
  int   l            = problem->l                    ;
  int   max_nr_class = 16                            ;
  int   nr_class     = 0                             ;
  int * label        = new int [ max_nr_class ]      ;
  int * count        = new int [ max_nr_class ]      ;
  int * data_label   = new int [ l            ]      ;
  ////////////////////////////////////////////////////
  nFullLoop ( i , l )                                {
    int this_label = (int)problem->y[i]              ;
    int j = 0                                        ;
    for ( j = 0 ; j < nr_class ; j++ )               {
      if ( this_label == label[j] )                  {
        ++count[j]                                   ;
        break                                        ;
      }                                              ;
    }                                                ;
    //////////////////////////////////////////////////
    data_label [ i ] = j                             ;
    if (j == nr_class)                               {
      if ( nr_class == max_nr_class )                {
        max_nr_class *= 2                            ;
        int * new_label                              ;
        int * new_count                              ;
        new_label = new int [ max_nr_class ]         ;
        new_count = new int [ max_nr_class ]         ;
        memcpy(new_label,label,sizeof(int)*nr_class) ;
        memcpy(new_count,count,sizeof(int)*nr_class) ;
        nDeleteArray ( label )                       ;
        nDeleteArray ( count )                       ;
        label = new_label                            ;
        count = new_count                            ;
      }                                              ;
      label [ nr_class ] = this_label                ;
      count [ nr_class ] = 1                         ;
      ++nr_class                                     ;
    }                                                ;
  }                                                  ;
  ////////////////////////////////////////////////////
  int * start = new int [ nr_class ]                 ;
  start [ 0 ] = 0                                    ;
  for (int i=1;i<nr_class;i++)                       {
    start [ i ] = start [ i-1 ] + count [ i-1 ]      ;
  }                                                  ;
  nFullLoop ( i , l )                                {
    perm    [ start      [ data_label[i] ] ] = i     ;
    ++start [ data_label [ i             ] ]         ;
  }                                                  ;
  ////////////////////////////////////////////////////
  start [ 0 ] = 0                                    ;
  for (int i=1;i<nr_class;i++)                       {
    start[i] = start [ i-1 ] + count [ i-1 ]         ;
  }                                                  ;
  ////////////////////////////////////////////////////
  *nr_class_ret = nr_class                           ;
  *label_ret    = label                              ;
  *start_ret    = start                              ;
  *count_ret    = count                              ;
  ////////////////////////////////////////////////////
  nDeleteArray ( data_label )                        ;
}

//////////////////////////////////////////////////////////////////////////////

static void trainOne                        (
              N::NLP::LrProblem   * problem ,
              N::NLP::LrParameter * param   ,
              double              * w       ,
              double                Cp      ,
              double                Cn      )
{
  double eps = param->eps                                                      ;
  int    pos = 0                                                               ;
  int    neg = 0                                                               ;
  //////////////////////////////////////////////////////////////////////////////
  nFullLoop ( i , problem->l )                                                 {
    if ( problem->y[i] > 0 ) pos++                                             ;
  }                                                                            ;
  neg = problem->l - pos                                                       ;
  //////////////////////////////////////////////////////////////////////////////
  double primal_solver_tol = eps * MaxValue ( MinValue(pos,neg) , 1) / problem->l        ;
  //////////////////////////////////////////////////////////////////////////////
  N::NLP::LrFunction * fun_obj = NULL                                          ;
  switch ( param -> Solver )                                                   {
    case N::NLP::LrParameter::L2R_LR                                             : {
      double * C = new double[problem->l]                                      ;
      nFullLoop ( i , problem->l )                                             {
        if ( problem->y[i] > 0 ) C[i] = Cp                                     ;
                            else C[i] = Cn                                     ;
      }                                                                        ;
      fun_obj = new N::NLP::LrL2R ( problem , C )                                  ;
      N::NLP::LrTron tron_obj(fun_obj,primal_solver_tol)                           ;
      tron_obj.tron(w)                                                         ;
      delete fun_obj                                                           ;
      delete [] C                                                              ;
    }                                                                          ;
    break                                                                      ;
    case N::NLP::LrParameter::L2R_L2LOSS_SVC                                     : {
      double * C = new double[problem->l]                                      ;
      nFullLoop ( i , problem->l )                                             {
        if ( problem->y[i] > 0 ) C[i] = Cp                                     ;
                            else C[i] = Cn                                     ;
      }                                                                        ;
      fun_obj = new N::NLP::LrL2Svc ( problem , C )                                ;
      N::NLP::LrTron tron_obj(fun_obj,primal_solver_tol)                           ;
      tron_obj.tron(w)                                                         ;
      delete fun_obj                                                           ;
      delete [] C                                                              ;
    }                                                                          ;
    break                                                                      ;
    case N::NLP::LrParameter::L2R_L2LOSS_SVC_DUAL                                  :
      solve_l2r_l1l2_svc(problem,w,eps,Cp,Cn,N::NLP::LrParameter::L2R_L2LOSS_SVC_DUAL) ;
    break                                                                      ;
    case N::NLP::LrParameter::L2R_L1LOSS_SVC_DUAL                                  :
      solve_l2r_l1l2_svc(problem,w,eps,Cp,Cn,N::NLP::LrParameter::L2R_L1LOSS_SVC_DUAL) ;
    break                                                                      ;
    case N::NLP::LrParameter::L1R_L2LOSS_SVC                                     : {
      N::NLP::LrProblem prob_col                                                   ;
      N::NLP::LrNode  * x_space = NULL                                             ;
      transpose(problem,&x_space,&prob_col)                                    ;
      solve_l1r_l2_svc(&prob_col,w,primal_solver_tol,Cp,Cn)                    ;
      delete [] prob_col.y                                                     ;
      delete [] prob_col.x                                                     ;
      delete [] x_space                                                        ;
    }                                                                          ;
    break                                                                      ;
    case N::NLP::LrParameter::L1R_LR                                             : {
      N::NLP::LrProblem prob_col                                                   ;
      N::NLP::LrNode  * x_space = NULL                                             ;
      transpose(problem,&x_space,&prob_col)                                    ;
      solve_l1r_lr(&prob_col,w,primal_solver_tol,Cp,Cn)                        ;
      delete [] prob_col.y                                                     ;
      delete [] prob_col.x                                                     ;
      delete [] x_space                                                        ;
    }                                                                          ;
    break                                                                      ;
    case N::NLP::LrParameter::L2R_LR_DUAL                                          :
      solve_l2r_lr_dual(problem,w,eps,Cp,Cn)                                   ;
    break                                                                      ;
    case N::NLP::LrParameter::L2R_L2LOSS_SVR                                     : {
      double * C = new double [ problem->l ]                                   ;
      nFullLoop ( i , problem->l ) C[i] = param->C                             ;
      fun_obj = new N::NLP::LrL2Svr ( problem , C , param->p )                     ;
      N::NLP::LrTron tron_obj(fun_obj,param->eps)                                  ;
      tron_obj.tron(w)                                                         ;
      delete fun_obj                                                           ;
      delete C                                                                 ;
    }                                                                          ;
    break                                                                      ;
    case N::NLP::LrParameter::L2R_L1LOSS_SVR_DUAL                                  :
      solve_l2r_l1l2_svr(problem,w,param,N::NLP::LrParameter::L2R_L1LOSS_SVR_DUAL) ;
    break                                                                      ;
    case N::NLP::LrParameter::L2R_L2LOSS_SVR_DUAL                                  :
      solve_l2r_l1l2_svr(problem,w,param,N::NLP::LrParameter::L2R_L2LOSS_SVR_DUAL) ;
    break                                                                      ;
    default                                                                    :
    break                                                                      ;
  }                                                                            ;
}

//////////////////////////////////////////////////////////////////////////////

N::NLP::Linear:: Linear(void)
{
  NrClass   = 0    ;
  NrFeature = 0    ;
  bias      = 0    ;
  w         = NULL ;
  label     = NULL ;
}

N::NLP::Linear::~Linear(void)
{
}

N::NLP::Linear & N::NLP::Linear::Train(LrProblem & problem,LrParameter & param)
{
  int l      = problem.l                             ;
  int n      = problem.n                             ;
  int w_size = problem.n                             ;
  ////////////////////////////////////////////////////
  if ( problem.bias >= 0 ) NrFeature = n - 1         ;
                      else NrFeature = n             ;
  Parameter = param                                  ;
  bias      = problem.bias                           ;
  ////////////////////////////////////////////////////
  if (Parameter.isSVR())                             {
    w       = new double [ w_size ]                  ;
    NrClass = 2                                      ;
    label   = NULL                                   ;
    trainOne(&problem,&Parameter,w,0,0)              ;
    return ME                                        ;
  }                                                  ;
  ////////////////////////////////////////////////////
  int   nr_class                                     ;
  int * Label    = NULL                              ;
  int * start    = NULL                              ;
  int * count    = NULL                              ;
  int * perm     = new int [l]                       ;
  ////////////////////////////////////////////////////
  // group training data of the same class
  groupClasses                                       (
    &problem  , &nr_class , &Label                   ,
    &start    , &count    ,  perm                  ) ;
  ////////////////////////////////////////////////////
  NrClass = nr_class                                 ;
  Label   = new int [nr_class]                       ;
  nFullLoop ( i , nr_class ) Label[i] = label[i]     ;
  ////////////////////////////////////////////////////
  // calculate weighted C
  double * weighted_C = new double[nr_class]         ;
  nFullLoop ( i , nr_class)                          {
    weighted_C[i] = Parameter.C                      ;
  }                                                  ;
  ////////////////////////////////////////////////////
  nFullLoop ( i , Parameter.NrWeight )               {
    int j                                            ;
    for (j=0;j<nr_class;j++)                         {
      if (Parameter.WeightLabel[i]==label[j]) break  ;
    }                                                ;
    if ( j == nr_class )                             {
// fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
// Error , not found
    } else                                           {
      weighted_C[j] *= Parameter.Weight[i]           ;
    }                                                ;
  }                                                  ;
  ////////////////////////////////////////////////////
  // constructing the subproblem
  LrNode ** x = new LrNode * [l]                     ;
  nFullLoop ( i , l )                                {
    x[i] = problem.x[ perm[i] ]                      ;
  }                                                  ;
  ////////////////////////////////////////////////////
  int k                                              ;
  LrProblem sub_prob                                 ;
  sub_prob.l = l                                     ;
  sub_prob.n = n                                     ;
  sub_prob.x = new LrNode * [sub_prob.l]             ;
  sub_prob.y = new double   [sub_prob.l]             ;
  ////////////////////////////////////////////////////
  nFullLoop ( i , sub_prob.l ) sub_prob.x[i]=x[i]    ;
  ////////////////////////////////////////////////////
  // multi-class svm by Crammer and Singer
  if (Parameter.Solver==LrParameter::MCSVM_CS)       {
    w = new double [ n * nr_class ]                  ;
    nFullLoop ( i , nr_class )                       {
      for (int j=start[i];j<start[i]+count[i];j++)   {
        sub_prob.y[j] = i                            ;
      }                                              ;
    }                                                ;
    LrSolver Solver                                  (
                   &sub_prob  , nr_class             ,
                   weighted_C , Parameter.eps      ) ;
    Solver . Solve ( w )                             ;
  } else                                             {
    if ( nr_class == 2 )                             {
      w = new double [ w_size ]                      ;
      ////////////////////////////////////////////////
      int e0 = start [ 0 ] + count [ 0 ]             ;
      k = 0                                          ;
      for (;k<e0        ;k++) sub_prob.y[k] = +1     ;
      for (;k<sub_prob.l;k++) sub_prob.y[k] = -1     ;
      trainOne(&sub_prob , &Parameter , w            ,
               weighted_C[0] , weighted_C[1]       ) ;
    } else                                           {
      double * W = new double [ w_size            ]  ;
      w          = new double [ w_size * nr_class ]  ;
      nFullLoop ( i , nr_class )                     {
        int si = start[i]                            ;
        int ei = count[i] + si                       ;
        //////////////////////////////////////////////
        k = 0                                        ;
        for (;k<si        ;k++) sub_prob.y[k] = -1   ;
        for (;k<ei        ;k++) sub_prob.y[k] = +1   ;
        for (;k<sub_prob.l;k++) sub_prob.y[k] = -1   ;
        trainOne(&sub_prob                           ,
                 &Parameter                          ,
                  W                                  ,
                  weighted_C[i]                      ,
                  Parameter.C                      ) ;
        nFullLoop ( j , w_size )                     {
          w [ j * nr_class + i ] = W[j]              ;
        }                                            ;
      }                                              ;
      nDeleteArray ( W )                             ;
    }                                                ;
  }                                                  ;
  ////////////////////////////////////////////////////
  nDeleteArray ( x          )                        ;
  nDeleteArray ( Label      )                        ;
  nDeleteArray ( start      )                        ;
  nDeleteArray ( count      )                        ;
  nDeleteArray ( perm       )                        ;
  nDeleteArray ( sub_prob.x )                        ;
  nDeleteArray ( sub_prob.y )                        ;
  nDeleteArray ( weighted_C )                        ;
  ////////////////////////////////////////////////////
  return ME                                          ;
}

bool N::NLP::Linear::ProbabilityModel(void)
{
  return Parameter . ProbabilityModel ( ) ;
}

void N::NLP::Linear::CrossValidation (
       LrProblem   & problem         ,
       LrParameter & param           ,
       int           nr_fold         ,
       double      * target          )
{
  int   l          = problem.l                                ;
  int * fold_start = new int [ nr_fold + 1 ]                  ;
  int * perm       = new int [ l           ]                  ;
  /////////////////////////////////////////////////////////////
  nFullLoop ( i , l ) perm [ i ] = i                          ;
  nFullLoop ( i , l )                                         {
    int j = i + rand() % (l-i)                                ;
    SwapValue ( perm[i] , perm[j] )                                ;
  }                                                           ;
  /////////////////////////////////////////////////////////////
  for (int i=0;i<=nr_fold;i++)                                {
    fold_start[i] = i * l / nr_fold                           ;
  }                                                           ;
  /////////////////////////////////////////////////////////////
  nFullLoop ( i , nr_fold )                                   {
    int begin = fold_start [ i     ]                          ;
    int end   = fold_start [ i + 1 ]                          ;
    int j,k                                                   ;
    LrProblem subprob                                         ;
    ///////////////////////////////////////////////////////////
    subprob.bias = problem.bias                               ;
    subprob.n    = problem.n                                  ;
    subprob.l    = l - ( end - begin )                        ;
    subprob.x    = new N::NLP::LrNode * [subprob.l]           ;
    subprob.y    = new double           [subprob.l]           ;
    ///////////////////////////////////////////////////////////
    k = 0                                                     ;
    for ( j=0 ; j < begin ; j++ )                             {
      subprob.x[k] = problem.x[perm[j]]                       ;
      subprob.y[k] = problem.y[perm[j]]                       ;
      ++k                                                     ;
    }                                                         ;
    for ( j=end ; j<l ; j++ )                                 {
      subprob.x[k] = problem.x[perm[j]]                       ;
      subprob.y[k] = problem.y[perm[j]]                       ;
      ++k                                                     ;
    }                                                         ;
    Linear submodel                                           ;
    submodel.Train(subprob,param)                             ;
    for ( j=begin ; j<end ; j++ )                             {
      target[perm[j]] = submodel.Predict(*problem.x[perm[j]]) ;
    }                                                         ;
    nDeleteArray ( subprob.x )                                ;
    nDeleteArray ( subprob.y )                                ;
  }                                                           ;
  /////////////////////////////////////////////////////////////
  nDeleteArray ( fold_start )                                 ;
  nDeleteArray ( perm       )                                 ;
}

double N::NLP::Linear::PredictValues(LrNode & x,double * decValues)
{
  int idx                                                            ;
  int n                                                              ;
  ////////////////////////////////////////////////////////////////////
  if ( bias >= 0 ) n = NrFeature + 1                                 ;
              else n = NrFeature                                     ;
  ////////////////////////////////////////////////////////////////////
  double * W        = w                                              ;
  int      nr_class = NrClass                                        ;
  int      nr_w                                                      ;
  if ( nr_class == 2 && Parameter.Solver!=LrParameter::MCSVM_CS)     {
    nr_w = 1                                                         ;
  } else nr_w = nr_class                                             ;
  ////////////////////////////////////////////////////////////////////
  LrNode * lx = &x                                                   ;
  nFullLoop ( i , nr_w ) decValues [ i ] = 0                         ;
  for (;(idx=lx->index)!=-1; lx++)                                   {
    // the dimension of testing data may exceed that of training
    if ( idx <= n )                                                  {
      nFullLoop ( i , nr_w )                                         {
        decValues[i] += W [ ( idx - 1 ) * nr_w + i ] * lx->value     ;
      }                                                              ;
    }                                                                ;
  }                                                                  ;
  ////////////////////////////////////////////////////////////////////
  if ( nr_class == 2 )                                               {
    if (Parameter.isSVR()) return decValues[0] ;                  else
    return ( decValues[0] > 0 ) ? label[0] : label[1]                ;
  } else                                                             {
    int dec_max_idx = 0                                              ;
    for (int i=1;i<nr_class;i++)                                     {
      if (decValues[i] > decValues[dec_max_idx]) dec_max_idx = i     ;
    }                                                                ;
    return label[dec_max_idx]                                        ;
  }                                                                  ;
}

double N::NLP::Linear::Predict(LrNode & x)
{
  double * decValues = new double [ NrClass ]          ;
  double   Label     = PredictValues ( x , decValues ) ;
  nDeleteArray ( decValues )                           ;
  return Label                                         ;
}

double N::NLP::Linear::PredictProbability(LrNode & x,double * probEstimates)
{
  nKickOut ( !ProbabilityModel() , 0 )                 ;
  //////////////////////////////////////////////////////
  int i                                                ;
  int nr_class = NrClass                               ;
  int nr_w                                             ;
  if ( nr_class == 2 ) nr_w = 1                        ;
                  else nr_w = nr_class                 ;
  //////////////////////////////////////////////////////
  double Label                                         ;
  Label = PredictValues ( x , probEstimates )          ;
  nFullLoop ( i , nr_w )                               {
    probEstimates[i] = 1 / (1+exp(-probEstimates[i]))  ;
  }                                                    ;
  //////////////////////////////////////////////////////
  if ( nr_class == 2 )                                 {
    // for binary classification
    probEstimates [ 1 ] = 1.0 - probEstimates[0]       ;
  } else                                               {
    double sum = 0                                     ;
    nFullLoop ( i , nr_class ) sum += probEstimates[i] ;
    nFullLoop ( i , nr_class ) probEstimates[i] /= sum ;
  }                                                    ;
  return Label                                         ;
}
