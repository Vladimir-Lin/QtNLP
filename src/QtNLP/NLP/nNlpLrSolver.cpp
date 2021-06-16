#include <qtnlp.h>

// A coordinate descent algorithm for
// multi-class support vector machines by Crammer and Singer
//
//  min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
//    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
//
//  where e^m_i = 0 if y_i  = m,
//        e^m_i = 1 if y_i != m,
//  C^m_i = C if m  = y_i,
//  C^m_i = 0 if m != y_i,
//  and w_m(\alpha) = \sum_i \alpha^m_i x_i
//
// Given:
// x, y, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Appendix of LIBLINEAR paper, Fan et al. (2008)

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
#define GETI(i) ((int)problem->y[i])
// To support weights for instances, use GETI(i) (i)

N::NLP::LrSolver:: LrSolver(LrProblem * Problem  ,
                            int         nr_class ,
                            double    * c        ,
                            double      EPS      ,
                            int         max_iter )
{
  wSize   = Problem->n          ;
  l       = Problem->l          ;
  nrClass = nr_class            ;
  eps     = EPS                 ;
  maxIter = max_iter            ;
  problem = Problem             ;
  B       = new double[nrClass] ;
  G       = new double[nrClass] ;
  C       = c                   ;
}

N::NLP::LrSolver::~LrSolver(void)
{
  if (NotNull(B)) delete [] B ;
  if (NotNull(G)) delete [] G ;
  B = NULL                    ;
  G = NULL                    ;
}

int nlp_compare_double(const void *a, const void *b)
{
  if ((*(double *)a) > (*(double *)b)) return -1 ;
  if ((*(double *)a) < (*(double *)b)) return  1 ;
  return 0                                       ;
}

void N::NLP::LrSolver::subSolve (
       double   A_i         ,
       int      yi          ,
       double   C_yi        ,
       int      active_i    ,
       double * alpha_new   )
{
  int      r                                                         ;
  double * D = new double[active_i]                                  ;
  ////////////////////////////////////////////////////////////////////
  nFullLoop ( i , active_i ) D[i] = B[i]                             ;
  if ( yi < active_i ) D[yi] += A_i * C_yi                           ;
  qsort(D,active_i,sizeof(double),nlp_compare_double)                ;
  ////////////////////////////////////////////////////////////////////
  double beta = D[0] - A_i * C_yi                                    ;
  for ( r = 1 ; r<active_i && beta<r*D[r] ; r++ ) beta += D[r]       ;
  beta /= r                                                          ;
  ////////////////////////////////////////////////////////////////////
  nFullLoop ( r , active_i )                                         {
    if (r == yi) alpha_new[r] = MinValue( C_yi      , (beta - B[r])/A_i ) ;
            else alpha_new[r] = MinValue( (double)0 , (beta - B[r])/A_i ) ;
  }                                                                  ;
  delete [] D                                                        ;
}

bool N::NLP::LrSolver::beShrunk(int i,int m,int yi,double alpha_i,double minG)
{
  double bound = 0                                     ;
  if (m == yi) bound = C[GETI(i)]                      ;
  if ((alpha_i == bound) && (G[m] < minG)) return true ;
  return false                                         ;
}

void N::NLP::LrSolver::Solve(double * w)
{
  int      iter           = 0                         ;
  int      active_size    = l                         ;
  int      lnrClass       = nrClass * l               ;
  int      wnrClass       = nrClass * wSize           ;
  double * alpha          = new double[ lnrClass  ]   ;
  double * alpha_new      = new double[ nrClass   ]   ;
  int    * index          = new int   [ l         ]   ;
  double * QD             = new double[ l         ]   ;
  int    * d_ind          = new int   [ nrClass   ]   ;
  double * d_val          = new double[ nrClass   ]   ;
  int    * alpha_index    = new int   [ lnrClass  ]   ;
  int    * y_index        = new int   [ l         ]   ;
  int    * active_size_i  = new int   [ l         ]   ;
  double   eps_shrink     = MaxValue( 10.0 * eps , 1.0 )   ; // stopping tolerance for shrinking
  bool     start_from_all = true                      ;
  /////////////////////////////////////////////////////
  // Initial alpha can be set here. Note that
  // sum_m alpha[i*nr_class+m] = 0, for all i=1,...,l-1
  // alpha[i*nr_class+m] <= C[GETI(i)] if prob->y[i] == m
  // alpha[i*nr_class+m] <= 0 if prob->y[i] != m
  // If initial alpha isn't zero, uncomment the for loop below to initialize w
  /////////////////////////////////////////////////////
  nFullLoop ( i , lnrClass ) alpha [ i ] = 0          ;
  nFullLoop ( i , wnrClass ) w     [ i ] = 0          ;
  nFullLoop ( i , l )                                 {
    nFullLoop ( m , nrClass )                         {
      alpha_index [ i * nrClass + m ] = m             ;
    }                                                 ;
    LrNode * xi = problem->x[i]                       ;
    QD[i] = 0                                         ;
    while ( xi->index != -1 )                         {
      double val = xi->value                          ;
      QD[i] += nSquare(val)                           ;
      // Uncomment the for loop if initial alpha isn't zero
      //  nFullLoop ( m , nrClass )                     {
      //    w [ (xi->index-1) * nrClass + m ] += alpha[ i * nrClass + m ] * val ;
      //  }                                             ;
      xi++                                            ;
    }                                                 ;
    active_size_i [ i ] = nrClass                     ;
    y_index [ i ] = (int)problem->y[i]                ;
    index   [ i ] = i                                 ;
  }                                                   ;
  /////////////////////////////////////////////////////
  while ( iter < maxIter )                            {
    double stopping = -INF                            ;
    nFullLoop ( i , active_size )                     {
      int j = i + rand()%(active_size-i)              ;
      SwapValue ( index[i] , index[j] )                    ;
    }                                                 ;
    ///////////////////////////////////////////////////
    nFullLoop ( s , active_size )                     {
      int      i       = index  [ s           ]       ;
      double   Ai      = QD     [ i           ]       ;
      double * alpha_i = &alpha [ i * nrClass ]       ;
      int    * alpha_index_i                          ;
      alpha_index_i = &alpha_index [ i * nrClass ]    ;
      /////////////////////////////////////////////////
      if ( Ai > 0 )                                   {
        nFullLoop ( m , active_size_i[i] ) G[m] = 1   ;
        if ( y_index[i] < active_size_i[i] )          {
          G [ y_index[i] ] = 0                        ;
        }                                             ;
        LrNode * xi = problem->x[i]                   ;
        while ( xi->index != -1)                      {
          double * w_i = &w[(xi->index-1)*nrClass]    ;
          nFullLoop ( m , active_size_i[i] )          {
            G[m] += w_i[alpha_index_i[m]]*(xi->value) ;
          }                                           ;
          xi++                                        ;
        }                                             ;
        ///////////////////////////////////////////////
        double minG =  INF                            ;
        double maxG = -INF                            ;
        ///////////////////////////////////////////////
        nFullLoop ( m , active_size_i[i] )            {
          if (alpha_i[alpha_index_i[m]] < 0          &&
              G[m] < minG) minG = G[m]                ;
          if (G[m] > maxG) maxG = G[m]                ;
        }                                             ;
        ///////////////////////////////////////////////
        if ( y_index[i] < active_size_i[i] )          {
            if (alpha_i[GETI(i)] < C[GETI(i)]        &&
                G[y_index[i]] < minG)                 {
            minG = G[ y_index[i] ]                    ;
          }                                           ;
        }                                             ;
        ///////////////////////////////////////////////
        nFullLoop ( m , active_size_i[i] )            {
          int yi = y_index       [ i ]                ;
          int mi = alpha_index_i [ m ]                ;
          if (beShrunk(i,m,yi,alpha_i[mi], minG))     {
            active_size_i[i]--                        ;
            while ( active_size_i[i] > m )            {
              int asi = active_size_i[i]              ;
              if (!beShrunk(i,asi,yi                  ,
                   alpha_i[ alpha_index_i[ asi ] ]    ,
                   minG)                            ) {
                SwapValue(alpha_index_i[m]                 ,
                     alpha_index_i[active_size_i[i]]) ;
                SwapValue(G[m], G[active_size_i[i]]      ) ;
                if (y_index[i] == active_size_i[i])   {
                  y_index[i] = m                      ;
                } else
                if (y_index[i] == m)                  {
                  y_index[i] = active_size_i[i]       ;
                }                                     ;
                break                                 ;
              }                                       ;
              active_size_i[i]--                      ;
            }                                         ;
          }                                           ;
        }                                             ;
        ///////////////////////////////////////////////
        if ( active_size_i[i] <= 1 )                  {
          active_size--                               ;
          SwapValue(index[s], index[active_size])          ;
          s--                                         ;
          continue                                    ;
        }                                             ;
        ///////////////////////////////////////////////
        if ( maxG-minG <= 1e-12 ) continue            ;
        else stopping = MaxValue(maxG - minG, stopping)    ;
        ///////////////////////////////////////////////
        nFullLoop ( m , active_size_i[i] )            {
          B[m] = G[m] - Ai*alpha_i[alpha_index_i[m]]  ;
        }                                             ;
        ///////////////////////////////////////////////
        subSolve                                      (
          Ai , y_index[i]  , C[GETI(i)]               ,
          active_size_i[i] , alpha_new              ) ;
        ///////////////////////////////////////////////
        int nz_d = 0                                  ;
        nFullLoop ( m , active_size_i[i] )            {
          double d                                    ;
          d = alpha_new[m]-alpha_i[alpha_index_i[m]]  ;
          alpha_i[alpha_index_i[m]] = alpha_new[m]    ;
          if ( fabs(d) >= 1e-12 )                     {
            d_ind[nz_d] = alpha_index_i[m]            ;
            d_val[nz_d] = d                           ;
            nz_d++                                    ;
          }                                           ;
        }                                             ;
        ///////////////////////////////////////////////
        xi = problem->x[i]                            ;
        while ( xi->index != -1 )                     {
          double * w_i = &w[(xi->index-1) * nrClass]  ;
          nFullLoop ( m , nz_d )                      {
            w_i[d_ind[m]] += d_val[m]*xi->value       ;
          }                                           ;
          xi++                                        ;
        }                                             ;
      }                                               ;
    }                                                 ;
    ///////////////////////////////////////////////////
    iter++                                            ;
    if ( stopping < eps_shrink )                      {
      if ( (stopping < eps) && start_from_all) break  ;
      else                                            {
        active_size = l                               ;
        nFullLoop ( i , l ) active_size_i[i]=nrClass  ;
        eps_shrink = MaxValue(eps_shrink/2,eps)            ;
        start_from_all = true                         ;
      }                                               ;
    } else start_from_all = false                     ;
  }                                                   ;
  /////////////////////////////////////////////////////
  // calculate objective value
  double v   = 0                                      ;
  int    nSV = 0                                      ;
  nFullLoop ( i , wnrClass ) v += nSquare ( w[i] )    ;
  v *= 0.5                                            ;
  /////////////////////////////////////////////////////
  nFullLoop ( i , lnrClass )                          {
    v += alpha[i]                                     ;
    if ( fabs(alpha[i]) > 0) nSV++                    ;
  }                                                   ;
  /////////////////////////////////////////////////////
  nFullLoop ( i , l )                                 {
    v -= alpha[ i * nrClass + (int)problem->y[i] ]    ;
  }                                                   ;
  /////////////////////////////////////////////////////
  delete [] alpha                                     ;
  delete [] alpha_new                                 ;
  delete [] index                                     ;
  delete [] QD                                        ;
  delete [] d_ind                                     ;
  delete [] d_val                                     ;
  delete [] alpha_index                               ;
  delete [] y_index                                   ;
  delete [] active_size_i                             ;
}
