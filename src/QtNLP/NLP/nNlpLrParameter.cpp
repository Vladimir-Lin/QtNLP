#include <qtnlp.h>

N::NLP::LrParameter:: LrParameter(void)
{
  Solver      = L2R_LR ;
  eps         = 0      ;
  C           = 0      ;
  NrWeight    = 0      ;
  p           = 0      ;
  WeightLabel = NULL   ;
  Weight      = NULL   ;
}

N::NLP::LrParameter:: LrParameter(const LrParameter & param)
{
  ME = param ;
}

N::NLP::LrParameter::~LrParameter(void)
{
  nDeleteArray ( WeightLabel ) ;
  nDeleteArray ( Weight      ) ;
}

N::NLP::LrParameter & N::NLP::LrParameter::operator = (const LrParameter & param)
{
  Solver      = param . Solver                                             ;
  eps         = param . eps                                                ;
  C           = param . C                                                  ;
  NrWeight    = param . NrWeight                                           ;
  p           = param . p                                                  ;
  WeightLabel = new int    [ NrWeight ]                                    ;
  Weight      = new double [ NrWeight ]                                    ;
  //////////////////////////////////////////////////////////////////////////
  memcpy ( WeightLabel , param . WeightLabel , sizeof(int   ) * NrWeight ) ;
  memcpy ( Weight      , param . Weight      , sizeof(double) * NrWeight ) ;
  return ME                                                                ;
}

bool N::NLP::LrParameter::isSVR(void)
{
  nKickOut ( Solver==L2R_L2LOSS_SVR      , true ) ;
  nKickOut ( Solver==L2R_L1LOSS_SVR_DUAL , true ) ;
  nKickOut ( Solver==L2R_L2LOSS_SVR_DUAL , true ) ;
  return false                                    ;
}

bool N::NLP::LrParameter::ProbabilityModel(void)
{
  nKickOut ( Solver==L2R_LR      , true ) ;
  nKickOut ( Solver==L2R_LR_DUAL , true ) ;
  nKickOut ( Solver==L1R_LR      , true ) ;
  return false                            ;
}

QString N::NLP::LrParameter::Check(void)
{
  if (eps <= 0) return "eps <= 0"      ;
  if (C   <= 0) return "C <= 0"        ;
  if (p   <  0) return "p < 0"         ;
  if (   Solver != L2R_LR
      && Solver != L2R_L2LOSS_SVC_DUAL
      && Solver != L2R_L2LOSS_SVC
      && Solver != L2R_L1LOSS_SVC_DUAL
      && Solver != MCSVM_CS
      && Solver != L1R_L2LOSS_SVC
      && Solver != L1R_LR
      && Solver != L2R_LR_DUAL
      && Solver != L2R_L2LOSS_SVR
      && Solver != L2R_L2LOSS_SVR_DUAL
      && Solver != L2R_L1LOSS_SVR_DUAL )
      return "unknown solver type"     ;
  return ""                            ;
}
