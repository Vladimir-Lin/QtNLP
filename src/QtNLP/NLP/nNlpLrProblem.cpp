#include <qtnlp.h>

N::NLP::LrProblem:: LrProblem(void)
{
}

N::NLP::LrProblem::~LrProblem(void)
{
}

void N::NLP::LrProblem::Camp(void)
{
  nFullLoop ( i , l )           {
    y[i] = (y[i] > 0) ? +1 : -1 ;
 }                              ;
}
